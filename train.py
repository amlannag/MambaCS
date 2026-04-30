"""
Training pipeline for DcTNN MRI reconstruction.

Define experiments in train_config.py, then run:
    python train.py --exp_idx <N>

All outputs are written to:
    {output_dir}/{prefix}_{name}/
        config.json       — config snapshot
        best_model.pth    — weights at best validation loss
        latest.pth        — latest checkpoint (resume from here)
        metrics.json      — per-epoch metrics
"""

import argparse
import dataclasses
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader

from DcTNN.tnn import cascadeNet, axVIT, patchVIT
from dc.dc import FFT_DC, fft_2d, ifft_2d
from dataset import MRIDataset, load_mask
from inference import run_inference
from config import Config
from train_config import EXPERIMENTS


def build_cfg(exp_idx: int) -> Config:
    cfg = Config()
    overrides = EXPERIMENTS[exp_idx]
    for key, val in overrides.items():
        if not hasattr(cfg, key):
            raise ValueError(f"Unknown config key '{key}' in EXPERIMENTS[{exp_idx}]")
        setattr(cfg, key, val)
    return cfg


_parser = argparse.ArgumentParser()
_parser.add_argument('--exp_idx', type=int, default=0)
_args = _parser.parse_args()
cfg = build_cfg(_args.exp_idx)
print(f"Experiment {_args.exp_idx}: {cfg.prefix}_{cfg.name}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def experiment_dir(cfg):
    folder = f"{cfg.prefix}_{cfg.name}"
    return os.path.join(cfg.output_dir, folder)


def psnr(pred, target, max_val=1.0):
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return torch.tensor(float('inf'))
    return 20.0 * torch.log10(torch.tensor(max_val, device=pred.device) / torch.sqrt(mse))


def config_to_dict(cfg):
    if dataclasses.is_dataclass(cfg):
        return {k: config_to_dict(v) for k, v in dataclasses.asdict(cfg).items()}
    return cfg


def append_metrics(path, record):
    history = []
    if os.path.exists(path):
        with open(path) as f:
            history = json.load(f)
    history.append(record)
    with open(path, 'w') as f:
        json.dump(history, f, indent=2)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

ENCODER_ARGS = {
    "axial": lambda cfg: (
        axVIT,
        dict(layerNo=cfg.layer_no, numCh=cfg.num_channels, d_model=None,
             nhead=cfg.nhead_axial, num_encoder_layers=cfg.num_encoder_layers,
             dim_feedforward=None, pos_emb_type=cfg.pos_emb_type,
             rope_theta=cfg.rope_theta, rope_mixed_rotate=cfg.rope_mixed_rotate)
    ),
    "kaleidoscope": lambda cfg: (
        patchVIT,
        dict(patch_size=cfg.patch_size, kaleidoscope=True, layerNo=cfg.layer_no,
             numCh=cfg.num_channels, nhead=cfg.nhead_patch,
             num_encoder_layers=cfg.num_encoder_layers,
             dim_feedforward=None, d_model=None, pos_emb_type=cfg.pos_emb_type,
             rope_theta=cfg.rope_theta, rope_mixed_rotate=cfg.rope_mixed_rotate)
    ),
    "patch": lambda cfg: (
        patchVIT,
        dict(patch_size=cfg.patch_size, kaleidoscope=False, layerNo=cfg.layer_no,
             numCh=cfg.num_channels, nhead=cfg.nhead_patch,
             num_encoder_layers=cfg.num_encoder_layers,
             dim_feedforward=None, d_model=None, pos_emb_type=cfg.pos_emb_type,
             rope_theta=cfg.rope_theta, rope_mixed_rotate=cfg.rope_mixed_rotate)
    ),
}


def build_model(cfg):
    enc_list, enc_args = [], []
    for name in cfg.encoders:
        if name not in ENCODER_ARGS:
            raise ValueError(f"Unknown encoder '{name}'. Choose from: {list(ENCODER_ARGS)}")
        cls, args = ENCODER_ARGS[name](cfg)
        enc_list.append(cls)
        enc_args.append(args)

    return cascadeNet(cfg.image_size, enc_list, enc_args, FFT_DC, cfg.learned_lambda)


# ---------------------------------------------------------------------------
# k-space simulation
# ---------------------------------------------------------------------------

def simulate_undersampling(gt_batch, mask, norm='ortho'):
    """
    gt_batch : [B, 1, N, N]
    mask     : [N, N]
    returns  : zf_image [B, 1, N, N],  kspace_us [B, 2, N, N]
    """
    kspace_full = fft_2d(gt_batch)
    kspace_us   = kspace_full * mask
    zf_image    = ifft_2d(kspace_us, norm=norm)[:, 0:cfg.num_channels, :, :]
    return zf_image, kspace_us


# ---------------------------------------------------------------------------
# Epoch helpers
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, masks, optimizer, criterion, device):
    model.train()
    mask_list  = list(masks.values())
    total_loss = 0.0

    for gt in loader:
        gt   = gt.to(device)
        mask = mask_list[np.random.randint(len(mask_list))].to(device)

        with torch.no_grad():
            zf_image, kspace_us = simulate_undersampling(gt, mask)

        optimizer.zero_grad()
        recon = model(zf_image, kspace_us, mask)
        loss  = criterion(recon, gt)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, masks, criterion, device):
    model.eval()
    mask_list  = list(masks.values())
    total_loss = 0.0
    total_psnr = 0.0

    for gt in loader:
        gt   = gt.to(device)
        mask = mask_list[np.random.randint(len(mask_list))].to(device)

        zf_image, kspace_us = simulate_undersampling(gt, mask)
        recon = model(zf_image, kspace_us, mask)

        total_loss += criterion(recon, gt).item()
        total_psnr += psnr(recon, gt).item()

    n = len(loader)
    return total_loss / n, total_psnr / n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---- Output directory ----
    out_dir = experiment_dir(cfg)
    os.makedirs(out_dir, exist_ok=True)

    metrics_path = os.path.join(out_dir, 'metrics.json')
    config_path  = os.path.join(out_dir, 'config.json')
    best_path    = os.path.join(out_dir, 'best_model.pth')
    latest_path  = os.path.join(out_dir, 'latest.pth')

    with open(config_path, 'w') as f:
        json.dump(config_to_dict(cfg), f, indent=2)

    wandb.init(
        project="MambaCS",
        name=f"{cfg.prefix}_{cfg.name}",
        config=config_to_dict(cfg),
    )

    print(f"Experiment : {cfg.prefix}_{cfg.name}")
    print(f"Encoders   : {cfg.encoders}")
    print(f"Output dir : {out_dir}")
    print(f"Device     : {device}")

    # ---- Masks ----
    masks = {}
    for R in cfg.acceleration_factors:
        path = os.path.join(cfg.mask_dir, f'mask_R{R}.png')
        if not os.path.exists(path):
            raise FileNotFoundError(f"Mask not found: {path}")
        masks[R] = load_mask(path, cfg.image_size)
    print(f"Masks      : R = {list(masks.keys())}")

    # ---- Datasets ----
    train_ds = MRIDataset(cfg.data_dir, N=cfg.image_size,
                          split='train', val_fraction=cfg.val_fraction,
                          seed=cfg.seed)
    val_ds   = MRIDataset(cfg.data_dir, N=cfg.image_size,
                          split='val',   val_fraction=cfg.val_fraction,
                          seed=cfg.seed)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              shuffle=True,  num_workers=cfg.num_workers,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size,
                              shuffle=False, num_workers=cfg.num_workers,
                              pin_memory=True)

    print(f"Train / Val: {len(train_ds)} / {len(val_ds)} samples")

    # ---- Model ----
    model    = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters : {n_params:,}")

    # ---- Optimiser / scheduler / loss ----
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=cfg.lr,
                                 weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.epochs,
        eta_min=cfg.lr * 1e-2,
    )
    criterion = nn.L1Loss()

    # ---- Resume ----
    start_epoch   = 0
    best_val_loss = float('inf')

    if cfg.resume and os.path.exists(cfg.resume):
        ckpt = torch.load(cfg.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch   = ckpt['epoch'] + 1
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        print(f"Resumed from epoch {start_epoch}  ({cfg.resume})")

    # ---- Training loop ----
    print()
    for epoch in range(start_epoch, cfg.epochs):
        t0 = time.time()

        train_loss         = train_one_epoch(model, train_loader, masks,
                                             optimizer, criterion, device)
        val_loss, val_psnr = validate(model, val_loader, masks, criterion, device)
        scheduler.step()

        lr      = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0

        print(f"Epoch {epoch+1:03d}/{cfg.epochs}  |  "
              f"Train L1: {train_loss:.4f}  |  "
              f"Val L1: {val_loss:.4f}  |  "
              f"Val PSNR: {val_psnr:.2f} dB  |  "
              f"LR: {lr:.2e}  |  "
              f"{elapsed:.1f}s")

        metrics = {
            'epoch':      epoch + 1,
            'train_loss': round(train_loss, 6),
            'val_loss':   round(val_loss,   6),
            'val_psnr':   round(val_psnr,   4),
            'lr':         lr,
            'time_s':     round(elapsed, 1),
        }
        if model.lamb is not False:
            for i, lv in enumerate(model.lamb):
                metrics[f'lambda_{i}'] = round(lv.item(), 6)
        append_metrics(metrics_path, metrics)
        wandb.log(metrics, step=epoch + 1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch':         epoch,
                'model':         model.state_dict(),
                'optimizer':     optimizer.state_dict(),
                'scheduler':     scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'val_psnr':      val_psnr,
            }, best_path)
            print(f"  -> Best model saved  (val_loss={best_val_loss:.4f})")

        torch.save({
            'epoch':         epoch,
            'model':         model.state_dict(),
            'optimizer':     optimizer.state_dict(),
            'scheduler':     scheduler.state_dict(),
            'best_val_loss': best_val_loss,
        }, latest_path)

    wandb.finish()
    print(f"\nTraining complete.  Outputs saved to: {out_dir}")

    for R in [4, 6, 8]:
        run_inference(out_dir, num_images=5, accel=R, split='val')


if __name__ == '__main__':
    main()
