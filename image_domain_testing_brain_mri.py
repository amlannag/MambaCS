"""
Standalone training script for image-domain MRI reconstruction on the OASIS Brain MRI dataset.

Pipeline:
  PNG image [0,1]
    → FFT → full k-space
    → × mask → undersampled k-space  (DC reference)
    → IFFT → zero-filled image       (model input)
    → cascadeNet(learning="image")
    → L1 loss vs ground truth PNG
"""

import argparse
import json
import os
import time
from glob import glob

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from DcTNN.dc import fft_2d, ifft_2d
from train_utils import build_model


# ---------------------------------------------------------------------------
# GPU-native mask generator
# ---------------------------------------------------------------------------

class GPUMaskGenerator:
    """
    Generates 1D random column undersampling masks entirely on GPU.
    Always samples a central ACS (auto-calibration signal) region,
    then randomly samples outer columns to hit the target acceleration.
    """
    _DEFAULT_CENTER = {4: 0.08, 8: 0.04}

    def __init__(self, accelerations, center_fractions=None):
        self.accelerations = [int(a) for a in accelerations]
        if center_fractions is None:
            center_fractions = [self._DEFAULT_CENTER.get(a, 0.04) for a in self.accelerations]
        self.center_fractions = {int(a): float(c) for a, c in zip(accelerations, center_fractions)}

    def apply(self, kspace_full, accel, seed=None):
        """
        Args:
            kspace_full : [B, 1, H, W] complex64 on any device
            accel       : int acceleration factor
        Returns:
            kspace_us   : [B, 1, H, W] complex64 — undersampled k-space
            mask        : [1, 1, W] float32       — 1D column mask (broadcast-ready)
        """
        device = kspace_full.device
        W = kspace_full.shape[-1]
        cf = self.center_fractions[int(accel)]

        num_center = max(1, int(round(W * cf)))
        num_total  = max(num_center, W // int(accel))

        mask = torch.zeros(W, device=device)

        # Central ACS region
        c0 = (W - num_center) // 2
        mask[c0:c0 + num_center] = 1.0

        # Random outer columns
        num_outer = num_total - num_center
        if num_outer > 0:
            outer = (mask == 0).nonzero(as_tuple=True)[0]
            g = torch.Generator(device=device)
            if seed is not None:
                g.manual_seed(hash(seed) & 0xFFFFFFFF)
            perm = torch.randperm(len(outer), device=device, generator=g)
            mask[outer[perm[:num_outer]]] = 1.0

        mask = mask.view(1, 1, 1, W)  # broadcast over [B, 1, H, W]
        kspace_us = kspace_full * mask
        return kspace_us, mask, None  # mask: [1, 1, 1, W]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class OASISDataset(Dataset):
    """Loads OASIS brain MRI PNG slices as normalised [1,H,W] float32 tensors."""

    def __init__(self, data_dir, image_size=(256, 256)):
        self.paths = sorted(glob(os.path.join(data_dir, '*.png')))
        if not self.paths:
            raise ValueError(f"No PNG files found in {data_dir}")
        self.image_size = image_size  # (H, W)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('L')
        img = img.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)  # [1, H, W]


# ---------------------------------------------------------------------------
# Undersampling
# ---------------------------------------------------------------------------

def simulate_undersampling_png(img_batch, mask_generator, accel, seed=None):
    """
    Convert a batch of real-valued PNG images into model inputs for image-domain learning.
    Everything runs on the same device as img_batch.

    Returns:
        model_input : [B, 1, H, W] float32   — zero-filled magnitude (encoder input)
        DC_input    : [B, 1, H, W] complex64  — undersampled k-space (DC reference)
        gt          : [B, 1, H, W] float32    — ground truth magnitude
        mask        : [1, 1, W]   float32     — 1D column sampling mask
    """
    img_complex = img_batch.to(torch.complex64)
    kspace_full = fft_2d(img_complex)                    # [B,1,H,W] complex64 — on GPU

    kspace_us, mask, _ = mask_generator.apply(kspace_full, accel, seed=seed)

    img_us      = ifft_2d(kspace_us)
    model_input = torch.abs(img_us)                      # zero-filled magnitude
    DC_input    = kspace_us                              # measured k-space
    gt          = img_batch

    return model_input, DC_input, gt, mask


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def psnr(pred, target):
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return torch.tensor(float('inf'))
    max_val = target.max().clamp(min=1e-8)
    return 20.0 * torch.log10(max_val / torch.sqrt(mse))


# ---------------------------------------------------------------------------
# Epoch helpers
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, mask_generator, accel_factors, optimizer, device):
    model.train()
    total_loss = 0.0
    total_psnr = 0.0

    for img_batch in loader:
        img_batch = img_batch.to(device)
        R = int(accel_factors[np.random.randint(len(accel_factors))])

        model_input, DC_input, gt, mask = simulate_undersampling_png(
            img_batch, mask_generator, R)

        optimizer.zero_grad()
        recon = model(model_input, DC_input, mask)
        loss  = F.l1_loss(recon, gt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        with torch.no_grad():
            total_loss += loss.item()
            total_psnr += psnr(recon.detach(), gt).item()

    n = len(loader)
    return total_loss / n, total_psnr / n


@torch.no_grad()
def validate(model, loader, mask_generator, accel_factors, device, seed_base=42):
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    total_zf_psnr = 0.0

    for batch_idx, img_batch in enumerate(loader):
        img_batch = img_batch.to(device)
        R = int(accel_factors[np.random.randint(len(accel_factors))])

        model_input, DC_input, gt, mask = simulate_undersampling_png(
            img_batch, mask_generator, R,
            seed=(seed_base, batch_idx, R))

        recon = model(model_input, DC_input, mask)

        total_loss    += F.l1_loss(recon, gt).item()
        total_psnr    += psnr(recon, gt).item()
        total_zf_psnr += psnr(model_input, gt).item()

    n = len(loader)
    return total_loss / n, total_psnr / n, total_zf_psnr / n


# ---------------------------------------------------------------------------
# Config shim — lets build_model() work without a full Config object
# ---------------------------------------------------------------------------

class _Cfg:
    image_size      = (256, 256)
    num_channels    = 1
    encoders        = ["axial", "axial", "axial"]
    patch_size      = (16, 16)
    nhead_patch     = 8
    nhead_axial     = 8
    layer_no        = 1
    num_encoder_layers = 2
    learning        = "image"
    lambda_schedule = "none"        # "none" → learned lambda per stage
    pos_emb_type    = "APE"
    attn_type       = "standard"
    rope_theta      = 100.0
    rope_mixed_rotate = True
    axial_row_stride  = 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Image-domain brain MRI training (OASIS)")
    parser.add_argument('--train_dir', default='/scratch/user/uqanag/OASIS/keras_png_slices_train')
    parser.add_argument('--val_dir',   default='/scratch/user/uqanag/OASIS/keras_png_slices_validate')
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--encoders',   nargs='+', default=['axial', 'axial', 'axial'])
    parser.add_argument('--accel',      nargs='+', type=int, default=[4])
    parser.add_argument('--epochs',     type=int,   default=100)
    parser.add_argument('--batch_size', type=int,   default=16)
    parser.add_argument('--lr',         type=float, default=1e-4)
    parser.add_argument('--num_workers',type=int,   default=4)
    parser.add_argument('--out_dir',    default='../Experiments/oasis_image_domain_4x')
    parser.add_argument('--wandb_project', default='MambaCS-OASIS')
    parser.add_argument('--resume',     default=None)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_size = (args.image_size, args.image_size)

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    # ---- Config shim ----
    cfg = _Cfg()
    cfg.image_size   = image_size
    cfg.encoders     = args.encoders

    # ---- Datasets ----
    train_ds = OASISDataset(args.train_dir, image_size=image_size)
    val_ds   = OASISDataset(args.val_dir,   image_size=image_size)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              persistent_workers=args.num_workers > 0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True,
                              persistent_workers=args.num_workers > 0)

    print(f"Train : {len(train_ds)} images  |  Val : {len(val_ds)} images")

    # ---- Mask generator ----
    mask_generator = GPUMaskGenerator(args.accel)

    # ---- Model ----
    model    = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model : {cfg.encoders}  |  {n_params:,} parameters  |  device={device}")

    # ---- Optimiser ----
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 1e-2)

    # ---- Resume ----
    start_epoch   = 0
    best_val_loss = float('inf')
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch   = ckpt['epoch'] + 1
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        print(f"Resumed from epoch {start_epoch}  ({args.resume})")

    # ---- WandB ----
    wandb.init(
        project=args.wandb_project,
        name=f"image_axial_{args.accel}x_oasis",
        config=vars(args),
    )

    metrics_path = os.path.join(args.out_dir, 'metrics.json')
    best_path    = os.path.join(args.out_dir, 'best_model.pth')
    latest_path  = os.path.join(args.out_dir, 'latest.pth')

    # ---- Training loop ----
    print()
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        train_loss, train_psnr = train_one_epoch(
            model, train_loader, mask_generator, args.accel, optimizer, device)
        val_loss, val_psnr, val_zf_psnr = validate(
            model, val_loader, mask_generator, args.accel, device)
        scheduler.step()

        lr      = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0

        print(f"Epoch {epoch+1:03d}/{args.epochs}  |  "
              f"Train L1: {train_loss:.6f}  Train PSNR: {train_psnr:.2f} dB  |  "
              f"Val L1: {val_loss:.6f}  Val PSNR: {val_psnr:.2f} dB  (ZF: {val_zf_psnr:.2f} dB)  |  "
              f"LR: {lr:.2e}  |  {elapsed:.1f}s")

        metrics = {
            'epoch':       epoch + 1,
            'train_l1':    round(train_loss,   6),
            'train_psnr':  round(train_psnr,   4),
            'val_l1':      round(val_loss,     6),
            'val_psnr':    round(val_psnr,     4),
            'val_zf_psnr': round(val_zf_psnr,  4),
            'lr':          lr,
            'time_s':      round(elapsed, 1),
        }
        history = []
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                history = json.load(f)
        history.append(metrics)
        with open(metrics_path, 'w') as f:
            json.dump(history, f, indent=2)
        wandb.log(metrics, step=epoch + 1)

        ckpt = {
            'epoch':         epoch,
            'model':         model.state_dict(),
            'optimizer':     optimizer.state_dict(),
            'scheduler':     scheduler.state_dict(),
            'best_val_loss': best_val_loss,
        }
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({**ckpt, 'best_val_loss': best_val_loss, 'val_psnr': val_psnr}, best_path)
            print(f"  -> Best model saved  (val_l1={best_val_loss:.6f})")
        torch.save(ckpt, latest_path)

    wandb.finish()
    print(f"\nTraining complete.  Outputs saved to: {args.out_dir}")


if __name__ == '__main__':
    main()
