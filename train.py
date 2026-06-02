"""
Training pipeline for Mamba Compressed Sensing MRI reconstruction.
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

from DcTNN.model import cascadeNet, axVIT, TokenVIT
from DcTNN.dc import fft_2d, ifft_2d
from dataset import H5MRIDataset
from inference import run_inference
from config import Config
from train_config import EXPERIMENTS
from DcTNN.lambda_scheduler import LambdaScheduler


def build_cfg(exp_idx: int) -> Config:
    """
    Builds a config object by applying overrides from EXPERIMENTS[exp_idx] to the default Config."""
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
    """
    Calculate the Peak Signal-to-Noise Ratio (PSNR) between prediction and target.
    """
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return torch.tensor(float('inf'))
    return 20.0 * torch.log10(torch.tensor(max_val, device=pred.device) / torch.sqrt(mse))

def config_to_dict(cfg):
    """
    Convert a config object to a dictionary.
    This is used for saving the config as JSON and logging to wandb."""

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
# Model Construction Using Configs
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
        TokenVIT,
        dict(patch_size=cfg.patch_size, tokenizer_type="kaleidoscope", layerNo=cfg.layer_no,
             numCh=cfg.num_channels, nhead=cfg.nhead_patch,
             num_encoder_layers=cfg.num_encoder_layers,
             dim_feedforward=None, d_model=None, pos_emb_type=cfg.pos_emb_type,
             rope_theta=cfg.rope_theta, rope_mixed_rotate=cfg.rope_mixed_rotate)
    ),
    "patch": lambda cfg: (
        TokenVIT,
        dict(patch_size=cfg.patch_size, tokenizer_type="patch", layerNo=cfg.layer_no,
             numCh=cfg.num_channels, nhead=cfg.nhead_patch,
             num_encoder_layers=cfg.num_encoder_layers,
             dim_feedforward=None, d_model=None, pos_emb_type=cfg.pos_emb_type,
             rope_theta=cfg.rope_theta, rope_mixed_rotate=cfg.rope_mixed_rotate)
    ),
}


def build_model(cfg):
    numCh = 2 if cfg.k_space_learning else 1   # k-space=2ch, image=1ch

    enc_list, enc_args = [], []
    for name in cfg.encoders:
        if name not in ENCODER_ARGS:
            raise ValueError(f"Unknown encoder '{name}'. Choose from: {list(ENCODER_ARGS)}")
        cls, args = ENCODER_ARGS[name](cfg)
        args['numCh'] = numCh
        enc_list.append(cls)
        enc_args.append(args)

    use_learned_lamb = cfg.lambda_schedule == "none"
    return cascadeNet(cfg.image_size, enc_list, enc_args,
                      use_learned_lamb, k_space_learning=cfg.k_space_learning)

# ---------------------------------------------------------------------------
# k-space simulation
# ---------------------------------------------------------------------------
def simulate_undersampling(kspace_full, mask, k_space_learning=True):
    """
    kspace_full     : [B, 2, N, N]  fully sampled k-space (real+imag)
    mask            : [N, N]
    k_space_learning: bool — controls domain of model_input and gt_norm

    Returns:
        model_input  [B, 2, N, N] norm undersampled k-space  (k_space=True)
                     [B, 1, N, N] norm zero-filled image      (k_space=False)
        kspace_norm  [B, 2, N, N] norm undersampled k-space  (DC reference, always)
        gt_norm      [B, 2, N, N] norm fully sampled k-space  (k_space=True)
                     [B, 1, N, N] norm fully sampled image    (k_space=False)
        norm_stats   dict: 'mean' [B,1,1,1], 'std' [B,1,1,1]
    """
    kspace_us = kspace_full * mask
    img_us    = ifft_2d(kspace_us)                        # [B, 2, N, N]
    real      = img_us[:, 0:1]
    mean      = real.mean(dim=(-2, -1), keepdim=True)
    std       = real.std(dim=(-2, -1), keepdim=True).clamp(min=1e-8)

    img_norm         = img_us.clone()
    img_norm[:, 0:1] = (real - mean) / std
    kspace_norm = fft_2d(img_norm)                        # [B, 2, N, N]

    if k_space_learning:
        model_input = kspace_norm
        img_gt_full         = ifft_2d(kspace_full)
        img_gt_norm         = img_gt_full.clone()
        img_gt_norm[:, 0:1] = (img_gt_full[:, 0:1] - mean) / std
        gt_norm = fft_2d(img_gt_norm)                     # [B, 2, N, N]
    else:
        model_input = img_norm[:, 0:1]                    # [B, 1, N, N]
        gt_norm = (ifft_2d(kspace_full)[:, 0:1] - mean) / std

    return model_input, kspace_norm, gt_norm, {'mean': mean, 'std': std}


def generate_column_mask(N, R, device):
    """Randomly sample N//R columns; returns a [N, N] float32 mask."""
    cols = torch.randperm(N, device=device)[:N // R]
    mask = torch.zeros(N, N, device=device)
    mask[:, cols] = 1.0
    return mask


# ---------------------------------------------------------------------------
# Epoch helpers
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, accel_factors, image_size, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for kspace_full in loader:
        kspace_full = kspace_full.to(device)
        R    = accel_factors[np.random.randint(len(accel_factors))]
        mask = generate_column_mask(image_size, R, device)

        with torch.no_grad():
            model_input, kspace_norm, gt_norm, _ = simulate_undersampling(
                kspace_full, mask, cfg.k_space_learning)

        optimizer.zero_grad()
        recon = model(model_input, kspace_norm, mask)
        loss  = criterion(recon, gt_norm)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, accel_factors, image_size, criterion, device):
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0

    for kspace_full in loader:
        kspace_full = kspace_full.to(device)
        R    = accel_factors[np.random.randint(len(accel_factors))]
        mask = generate_column_mask(image_size, R, device)

        model_input, kspace_norm, gt_norm, _ = simulate_undersampling(
            kspace_full, mask, cfg.k_space_learning)
        recon = model(model_input, kspace_norm, mask)

        total_loss += criterion(recon, gt_norm).item()
        total_psnr += psnr(recon, gt_norm).item()

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

    print(f"Accel      : R = {cfg.acceleration_factors}  ({cfg.image_size // cfg.acceleration_factors[0]} cols sampled for R={cfg.acceleration_factors[0]})")

    # ---- Datasets ----
    train_ds = H5MRIDataset(cfg.data_dir, N=cfg.image_size,
                            split='train', val_fraction=cfg.val_fraction,
                            seed=cfg.seed, kspace_key=cfg.kspace_key)
    val_ds   = H5MRIDataset(cfg.data_dir, N=cfg.image_size,
                            split='val',   val_fraction=cfg.val_fraction,
                            seed=cfg.seed, kspace_key=cfg.kspace_key)

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
    def criterion(recon, target):
        """Mean |recon - target|^2 over all elements (complex-aware squared loss)."""
        return ((recon - target) ** 2).mean()

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

    # ---- Lambda scheduler (if active) ----
    lamb_sched = None
    if cfg.lambda_schedule != "none":
        lamb_sched = LambdaScheduler(cfg.lambda_schedule, cfg.lambda_start,
                                     cfg.lambda_end, cfg.epochs)

    # ---- Training loop ----
    print()
    for epoch in range(start_epoch, cfg.epochs):
        t0 = time.time()

        if lamb_sched is not None:
            model.set_scheduled_lamb(lamb_sched.get_lambda(epoch))

        train_loss         = train_one_epoch(model, train_loader, cfg.acceleration_factors,
                                             cfg.image_size, optimizer, criterion, device)
        val_loss, val_psnr = validate(model, val_loader, cfg.acceleration_factors,
                                      cfg.image_size, criterion, device)
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
        elif lamb_sched is not None:
            metrics['lambda_scheduled'] = round(model.scheduled_lamb, 6)
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
