"""
Training pipeline for DcTNN MRI reconstruction.

Configure everything in train_config.py, then run:
    python train.py

All outputs are written to:
    {output_dir}/{prefix}_{name}/
        config.json       — config snapshot
        best_model.pth    — weights at best validation loss
        latest.pth        — latest checkpoint (resume from here)
        metrics.json      — per-epoch metrics
"""

import dataclasses
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from DcTNN.tnn import cascadeNet, axVIT, patchVIT
from dc.dc import FFT_DC, fft_2d, ifft_2d
from dataset import MRIDataset, load_mask
from train_config import cfg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def experiment_dir(cfg):
    folder = f"{cfg.experiment.prefix}_{cfg.experiment.name}"
    return os.path.join(cfg.experiment.output_dir, folder)


def psnr(pred, target, max_val=1.0):
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return torch.tensor(float('inf'))
    return 20.0 * torch.log10(torch.tensor(max_val, device=pred.device) / torch.sqrt(mse))


def config_to_dict(cfg):
    """Recursively convert nested dataclasses to a plain dict for JSON serialisation."""
    if dataclasses.is_dataclass(cfg):
        return {k: config_to_dict(v) for k, v in dataclasses.asdict(cfg).items()}
    return cfg


def append_metrics(path, record):
    """Append one epoch record to metrics.json (creates the file on first call)."""
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

def build_model(cfg):
    m = cfg.model
    d = cfg.data

    patch_args = dict(patch_size=m.patch_size, kaleidoscope=False, layerNo=m.layer_no,
                      numCh=d.num_channels, nhead=m.nhead_patch,
                      num_encoder_layers=m.num_encoder_layers,
                      dim_feedforward=None, d_model=None)
    kd_args    = dict(patch_size=m.patch_size, kaleidoscope=True,  layerNo=m.layer_no,
                      numCh=d.num_channels, nhead=m.nhead_patch,
                      num_encoder_layers=m.num_encoder_layers,
                      dim_feedforward=None, d_model=None)
    ax_args    = dict(layerNo=m.layer_no, numCh=d.num_channels, d_model=None,
                      nhead=m.nhead_axial,
                      num_encoder_layers=m.num_encoder_layers,
                      dim_feedforward=None)

    return cascadeNet(
        d.image_size,
        [axVIT,    patchVIT, patchVIT],
        [ax_args, kd_args,  patch_args],
        FFT_DC,
        m.learned_lambda,
    )


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
    zf_image    = ifft_2d(kspace_us, norm=norm)[:, 0:cfg.data.num_channels, :, :]
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
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.training.grad_clip)
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

    # Save config snapshot so the run is always reproducible
    with open(config_path, 'w') as f:
        json.dump(config_to_dict(cfg), f, indent=2)

    print(f"Experiment : {cfg.experiment.prefix}_{cfg.experiment.name}")
    print(f"Output dir : {out_dir}")
    print(f"Device     : {device}")

    # ---- Masks ----
    masks = {}
    for R in cfg.data.acceleration_factors:
        path = os.path.join(cfg.data.mask_dir, f'mask_R{R}.png')
        if not os.path.exists(path):
            raise FileNotFoundError(f"Mask not found: {path}")
        masks[R] = load_mask(path, cfg.data.image_size)
    print(f"Masks      : R = {list(masks.keys())}")

    # ---- Datasets ----
    train_ds = MRIDataset(cfg.data.data_dir, N=cfg.data.image_size,
                          split='train', val_fraction=cfg.data.val_fraction,
                          seed=cfg.data.seed)
    val_ds   = MRIDataset(cfg.data.data_dir, N=cfg.data.image_size,
                          split='val',   val_fraction=cfg.data.val_fraction,
                          seed=cfg.data.seed)

    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size,
                              shuffle=True,  num_workers=cfg.training.num_workers,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.training.batch_size,
                              shuffle=False, num_workers=cfg.training.num_workers,
                              pin_memory=True)

    print(f"Train / Val: {len(train_ds)} / {len(val_ds)} samples")

    # ---- Model ----
    model    = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters : {n_params:,}")

    # ---- Optimiser / scheduler / loss ----
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=cfg.training.lr,
                                 weight_decay=cfg.training.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.training.epochs,
        eta_min=cfg.training.lr * 1e-2,
    )
    criterion = nn.L1Loss()

    # ---- Resume ----
    start_epoch   = 0
    best_val_loss = float('inf')

    resume_path = cfg.experiment.resume
    if resume_path and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch   = ckpt['epoch'] + 1
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        print(f"Resumed from epoch {start_epoch}  ({resume_path})")

    # ---- Training loop ----
    print()
    for epoch in range(start_epoch, cfg.training.epochs):
        t0 = time.time()

        train_loss         = train_one_epoch(model, train_loader, masks,
                                             optimizer, criterion, device)
        val_loss, val_psnr = validate(model, val_loader, masks, criterion, device)
        scheduler.step()

        lr      = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0

        print(f"Epoch {epoch+1:03d}/{cfg.training.epochs}  |  "
              f"Train L1: {train_loss:.4f}  |  "
              f"Val L1: {val_loss:.4f}  |  "
              f"Val PSNR: {val_psnr:.2f} dB  |  "
              f"LR: {lr:.2e}  |  "
              f"{elapsed:.1f}s")

        # Append metrics for this epoch
        append_metrics(metrics_path, {
            'epoch':      epoch + 1,
            'train_loss': round(train_loss, 6),
            'val_loss':   round(val_loss,   6),
            'val_psnr':   round(val_psnr,   4),
            'lr':         lr,
            'time_s':     round(elapsed, 1),
        })

        # Save best weights
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

        # Save latest checkpoint (for resuming)
        torch.save({
            'epoch':         epoch,
            'model':         model.state_dict(),
            'optimizer':     optimizer.state_dict(),
            'scheduler':     scheduler.state_dict(),
            'best_val_loss': best_val_loss,
        }, latest_path)

    print(f"\nTraining complete.  Outputs saved to: {out_dir}")


if __name__ == '__main__':
    main()
