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

from dataset import H5MRIDataset
from config import Config
from train_config import EXPERIMENTS
from DcTNN.lambda_scheduler import LambdaScheduler
from train_utils import FastMRIMaskGenerator, build_model, resolve_data_dirs, simulate_undersampling
from loss import MagnitudeL1Loss
from DcTNN.dc import ifft_2d


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

def psnr(pred, target, max_val=None):
    """PSNR between pred and target. max_val defaults to target.max()."""
    mse = torch.mean(torch.abs(pred - target) ** 2)
    if mse == 0:
        return torch.tensor(float('inf'))
    mv = target.max() if max_val is None else torch.tensor(max_val, device=pred.device)
    return 20.0 * torch.log10(mv.to(pred.device) / torch.sqrt(mse))

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
# Epoch helpers
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, accel_factors, mask_generator, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_psnr = 0.0

    for kspace_full in loader:
        kspace_full = kspace_full.to(device)
        R    = accel_factors[np.random.randint(len(accel_factors))]
        kspace_us, mask, _ = mask_generator.apply(kspace_full, R)

        with torch.no_grad():
            model_input, DC_input, gt, _ = simulate_undersampling(
                kspace_full, mask, cfg.learning, cfg.norm, kspace_us=kspace_us)

        optimizer.zero_grad()
        recon = model(model_input, DC_input, mask)
        loss  = criterion(recon, gt)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
        optimizer.step()

        with torch.no_grad():
            recon_mag = torch.abs(ifft_2d(recon)) if recon.is_complex() else recon
            total_loss += loss.item()
            total_psnr += psnr(recon_mag, gt).item()

    n = len(loader)
    return total_loss / n, total_psnr / n


@torch.no_grad()
def validate(model, loader, accel_factors, image_size, criterion, device):
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    total_zf_psnr = 0.0

    mask_generator = FastMRIMaskGenerator(
        accel_factors,
        center_fractions=cfg.center_fractions,
        mask_type=cfg.mask_type,
    )

    for batch_idx, kspace_full in enumerate(loader):
        kspace_full = kspace_full.to(device)
        R    = accel_factors[np.random.randint(len(accel_factors))]
        kspace_us, mask, _ = mask_generator.apply(kspace_full, R, seed=(cfg.seed, batch_idx, int(R)))

        model_input, DC_input, gt, _ = simulate_undersampling(
            kspace_full, mask, cfg.learning, cfg.norm, kspace_us=kspace_us)
        recon = model(model_input, DC_input, mask)

        recon_mag = torch.abs(ifft_2d(recon)) if recon.is_complex() else recon
        zf_mag    = torch.abs(ifft_2d(DC_input)) if DC_input.is_complex() else model_input
        total_loss    += criterion(recon, gt).item()
        total_psnr    += psnr(recon_mag, gt).item()
        total_zf_psnr += psnr(zf_mag, gt).item()

    n = len(loader)
    return total_loss / n, total_psnr / n, total_zf_psnr / n


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

    img_h, img_w = cfg.image_size if isinstance(cfg.image_size, tuple) else (cfg.image_size, cfg.image_size)
    print(f"Accel      : R = {cfg.acceleration_factors}  |  mask={cfg.mask_type}  |  center_fractions={cfg.center_fractions}")

    # ---- Datasets ----
    train_data_dir, val_data_dir = resolve_data_dirs(cfg)
    train_ds = H5MRIDataset(train_data_dir, image_size=cfg.image_size,
                            kspace_key=cfg.kspace_key)
    val_ds   = H5MRIDataset(val_data_dir, image_size=cfg.image_size,
                            kspace_key=cfg.kspace_key,
                            max_files=cfg.max_val_files)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              shuffle=True,  num_workers=cfg.num_workers,
                              pin_memory=True,
                              persistent_workers=cfg.num_workers > 0)

    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size,
                            shuffle=False, num_workers=cfg.num_workers,
                            pin_memory=True,
                            persistent_workers=cfg.num_workers > 0)

    print(f"Train dir   : {train_data_dir}")
    print(f"Val dir     : {val_data_dir}")
    if cfg.max_val_files is not None:
        print(f"Val files   : {len(val_ds.h5_files)} capped")
    print(f"Train / Val : {len(train_ds)} / {len(val_ds)} samples")

    # ---- Model ----
    model    = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters : {n_params:,}")
    mask_generator = FastMRIMaskGenerator(
        cfg.acceleration_factors,
        center_fractions=cfg.center_fractions,
        mask_type=cfg.mask_type,
    )

    # ---- Optimiser / scheduler / loss ----
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=cfg.lr,
                                 weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.epochs,
        eta_min=cfg.lr * 1e-2,
    )
    criterion = MagnitudeL1Loss()

    # ---- Resume ----
    start_epoch    = 0
    best_val_loss  = float('inf')

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

        train_loss, train_psnr = train_one_epoch(model, train_loader, cfg.acceleration_factors,
                                                mask_generator, optimizer, criterion, device)
        val_loss, val_psnr, val_zf_psnr = validate(model, val_loader, cfg.acceleration_factors,
                                                    cfg.image_size, criterion, device)
        scheduler.step()

        lr      = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0

        print(f"Epoch {epoch+1:03d}/{cfg.epochs}  |  "
              f"Train L1: {train_loss:.6f}  Train PSNR: {train_psnr:.2f} dB  |  "
              f"Val L1: {val_loss:.6f}  Val PSNR: {val_psnr:.2f} dB  (ZF: {val_zf_psnr:.2f} dB)  |  "
              f"LR: {lr:.2e}  |  {elapsed:.1f}s")

        metrics = {
            'epoch':        epoch + 1,
            'train_l1':     round(train_loss,    6),
            'train_psnr':   round(train_psnr,    4),
            'val_l1':       round(val_loss,      6),
            'val_psnr':     round(val_psnr,      4),
            'val_zf_psnr':  round(val_zf_psnr,   4),
            'lr':           lr,
            'time_s':       round(elapsed, 1),
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
            print(f"  -> Best model saved  (val_l1={best_val_loss:.6f})")

        torch.save({
            'epoch':         epoch,
            'model':         model.state_dict(),
            'optimizer':     optimizer.state_dict(),
            'scheduler':     scheduler.state_dict(),
            'best_val_loss': best_val_loss,
        }, latest_path)

    wandb.finish()
    print(f"\nTraining complete.  Outputs saved to: {out_dir}")

if __name__ == '__main__':
    main()
