"""
Training script using TCS-main's exact architecture on the OASIS Brain MRI dataset.

Uses TCS-main's Post-LN transformers, standard nn.TransformerEncoderLayer,
and FFT_DC that returns real() — exactly as the original repository.

Two mask modes (--mask_type):
  tcs : Fixed 2D mask from TCS-main/masks/mask_R{accel}.png (same pattern every batch)
  gpu : Random 1D column mask generated per batch (DC at corner, TCS FFT convention)

Both modes use TCS-main's FFT_DC and fft_2d throughout — no MambaCS DC code.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# ── Pull TCS-main onto the path before any of its modules are imported ────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_SCRIPT_DIR, 'TCS-main'))

from DcTNN.tnn import cascadeNet, axVIT   # Post-LN, nn.TransformerEncoderLayer
from dc.dc import FFT_DC                 # real() output, no fftshift
from dc.dc import fft_2d as tcs_fft_2d  # [B,1,H,W] → [B,2,H,W], DC at corner
from dc.dc import ifft_2d as tcs_ifft_2d


# ---------------------------------------------------------------------------
# Dataset (inline — avoids importing from MambaCS DcTNN which would conflict)
# ---------------------------------------------------------------------------

class OASISDataset(Dataset):
    """Loads OASIS brain MRI PNG slices as normalised [1,H,W] float32 tensors."""

    def __init__(self, data_dir, image_size=(256, 256)):
        from glob import glob
        self.paths = sorted(glob(os.path.join(data_dir, '*.png')))
        if not self.paths:
            raise ValueError(f"No PNG files found in {data_dir}")
        self.image_size = image_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('L')
        img = img.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)  # [1, H, W]


# ---------------------------------------------------------------------------
# Mask utilities — TCS convention: DC at corner (no fftshift in FFT)
# ---------------------------------------------------------------------------

def load_tcs_mask(mask_path, device):
    """
    Load a 2D mask PNG in TCS-main convention.
    ifftshift moves the ACS from display-center to FFT-corner,
    matching torch.fft.fft2 which puts DC at [0,0].
    """
    mask_np = np.array(Image.open(mask_path).convert('L'), dtype=np.float32)
    mask_np = np.fft.ifftshift(mask_np)
    mask_np = mask_np / mask_np.max()
    mask = torch.tensor(mask_np, device=device)  # [H, W]
    print(f"  Loaded TCS mask: {mask_path}  shape={tuple(mask.shape)}  "
          f"sampled_frac={mask.mean():.4f}")
    return mask


def generate_random_column_mask_tcs(H, W, accel, center_fraction=0.04,
                                    device='cpu', seed=None):
    """
    Random 1D column mask in TCS-main FFT convention (ACS at corners).

    Equivalent to generating a center-ACS column mask and applying ifftshift:
    the low-frequency columns land at the edges of the W dimension, matching
    where torch.fft.fft2 (no shift) puts the DC component.

    Returns [H, W] float32 (same column pattern broadcast across all rows).
    """
    num_center = max(1, int(round(W * center_fraction)))
    num_total  = max(num_center, W // accel)

    mask = torch.zeros(W, device=device)

    # ACS at corners: ifftshift of [W//2-n//2 : W//2+n//2] lands at [0:n//2] ∪ [W-n//2:]
    half  = num_center // 2
    extra = num_center - 2 * half  # 1 when num_center is odd
    mask[:half + extra] = 1.0
    if half > 0:
        mask[W - half:] = 1.0

    # Random outer columns
    num_outer = num_total - num_center
    if num_outer > 0:
        outer = (mask == 0).nonzero(as_tuple=True)[0]
        g = torch.Generator(device=device)
        if seed is not None:
            g.manual_seed(int(seed) & 0xFFFFFFFF)
        perm = torch.randperm(len(outer), device=device, generator=g)
        mask[outer[perm[:num_outer]]] = 1.0

    return mask.unsqueeze(0).expand(H, -1).contiguous()  # [H, W]


# ---------------------------------------------------------------------------
# Undersampling — TCS convention throughout
# ---------------------------------------------------------------------------

def simulate_undersampling_tcs(img_batch, mask):
    """
    Convert real PNG images to TCS-main model inputs using TCS-main's FFT.

    Args:
        img_batch : [B, 1, H, W] float32 in [0, 1]
        mask      : [H, W] float32  (DC at corner, TCS convention)

    Returns:
        model_input : [B, 1, H, W] float32  — real part of zero-filled IFFT
        y           : [B, 2, H, W] float32  — undersampled k-space (real + imag channels)
        gt          : [B, 1, H, W] float32  — ground truth image
    """
    kspace_full = tcs_fft_2d(img_batch)          # [B, 2, H, W], DC at corner
    y           = kspace_full * mask              # mask [H,W] broadcasts → [B,2,H,W]
    model_input = tcs_ifft_2d(y)[:, 0:1, :, :]  # real part of zero-filled [B,1,H,W]
    return model_input, y, img_batch


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def psnr(pred, target):
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return torch.tensor(float('inf'))
    return 20.0 * torch.log10(target.max().clamp(min=1e-8) / torch.sqrt(mse))


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, mask, mask_type, accel, optimizer, device):
    model.train()
    total_loss = total_psnr = 0.0

    for img_batch in loader:
        img_batch = img_batch.to(device)
        _, _, H, W = img_batch.shape

        m = (generate_random_column_mask_tcs(H, W, accel, device=device)
             if mask_type == 'gpu' else mask)

        model_input, y, gt = simulate_undersampling_tcs(img_batch, m)

        optimizer.zero_grad()
        recon = model(model_input, y, m)
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
def validate(model, loader, mask, mask_type, accel, device, seed_base=42):
    model.eval()
    total_loss = total_psnr = total_zf_psnr = 0.0

    for batch_idx, img_batch in enumerate(loader):
        img_batch = img_batch.to(device)
        _, _, H, W = img_batch.shape

        m = (generate_random_column_mask_tcs(
                 H, W, accel, device=device,
                 seed=seed_base * 1000 + batch_idx)
             if mask_type == 'gpu' else mask)

        model_input, y, gt = simulate_undersampling_tcs(img_batch, m)
        recon = model(model_input, y, m)

        total_loss    += F.l1_loss(recon, gt).item()
        total_psnr    += psnr(recon, gt).item()
        total_zf_psnr += psnr(model_input, gt).item()

    n = len(loader)
    return total_loss / n, total_psnr / n, total_zf_psnr / n


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="TCS-main architecture training on OASIS (image-domain)"
    )
    parser.add_argument('--train_dir',     default='/scratch/user/uqanag/OASIS/keras_png_slices_train')
    parser.add_argument('--val_dir',       default='/scratch/user/uqanag/OASIS/keras_png_slices_validate')
    parser.add_argument('--image_size',    type=int,   default=256)
    parser.add_argument('--mask_type',     choices=['tcs', 'gpu'], required=True,
                        help='tcs: fixed 2D PNG mask  |  gpu: random 1D column mask per batch')
    parser.add_argument('--accel',         type=int,   default=8)
    parser.add_argument('--epochs',        type=int,   default=100)
    parser.add_argument('--batch_size',    type=int,   default=16)
    parser.add_argument('--lr',            type=float, default=1e-4)
    parser.add_argument('--num_workers',   type=int,   default=4)
    parser.add_argument('--out_dir',       default='../Experiments/tcs_oasis_8x')
    parser.add_argument('--wandb_project', default='MambaCS-OASIS')
    parser.add_argument('--resume',        default=None)
    args = parser.parse_args()

    device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_size = (args.image_size, args.image_size)

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    # ── Mask ──────────────────────────────────────────────────────────────────
    if args.mask_type == 'tcs':
        mask_path = os.path.join(
            _SCRIPT_DIR, 'TCS-main', 'masks', f'mask_R{args.accel}.png'
        )
        mask = load_tcs_mask(mask_path, device)   # [H, W], loaded once
    else:
        mask = None   # generated fresh each batch in train/validate

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds = OASISDataset(args.train_dir, image_size=image_size)
    val_ds   = OASISDataset(args.val_dir,   image_size=image_size)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              persistent_workers=args.num_workers > 0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True,
                              persistent_workers=args.num_workers > 0)

    print(f"Train: {len(train_ds)} images  |  Val: {len(val_ds)} images")

    # ── Model — TCS-main architecture ─────────────────────────────────────────
    N      = args.image_size
    axArgs = dict(layerNo=1, numCh=1, d_model=None, nhead=8,
                  num_encoder_layers=2, dim_feedforward=None)
    model  = cascadeNet(
        N,
        encList=[axVIT, axVIT, axVIT],
        encArgs=[axArgs.copy(), axArgs.copy(), axArgs.copy()],
        dcFunc=FFT_DC,
        lamb=True,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: TCS-main cascadeNet  3× axVIT  Post-LN  |  "
          f"{n_params:,} params  |  device={device}")

    # ── Optimiser + scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 1e-2)

    # ── Resume ────────────────────────────────────────────────────────────────
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

    # ── WandB ─────────────────────────────────────────────────────────────────
    wandb.init(
        project=args.wandb_project,
        name=f"tcs_arch_{args.accel}x_{args.mask_type}_mask",
        config={**vars(args), 'n_params': n_params, 'architecture': 'TCS-main'},
    )

    metrics_path = os.path.join(args.out_dir, 'metrics.json')
    best_path    = os.path.join(args.out_dir, 'best_model.pth')
    latest_path  = os.path.join(args.out_dir, 'latest.pth')

    # ── Training loop ─────────────────────────────────────────────────────────
    print()
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        train_loss, train_psnr = train_one_epoch(
            model, train_loader, mask, args.mask_type, args.accel, optimizer, device)
        val_loss, val_psnr, val_zf_psnr = validate(
            model, val_loader, mask, args.mask_type, args.accel, device)
        scheduler.step()

        lr      = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0

        print(f"Epoch {epoch+1:03d}/{args.epochs}  |  "
              f"Train L1: {train_loss:.6f}  PSNR: {train_psnr:.2f} dB  |  "
              f"Val L1: {val_loss:.6f}  PSNR: {val_psnr:.2f} dB  "
              f"(ZF: {val_zf_psnr:.2f} dB)  |  "
              f"LR: {lr:.2e}  |  {elapsed:.1f}s")

        row = {
            'epoch':       epoch + 1,
            'train_l1':    round(train_loss,  6),
            'train_psnr':  round(train_psnr,  4),
            'val_l1':      round(val_loss,    6),
            'val_psnr':    round(val_psnr,    4),
            'val_zf_psnr': round(val_zf_psnr, 4),
            'lr':          lr,
            'time_s':      round(elapsed, 1),
        }
        history = []
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                history = json.load(f)
        history.append(row)
        with open(metrics_path, 'w') as f:
            json.dump(history, f, indent=2)
        wandb.log(row, step=epoch + 1)

        ckpt = {
            'epoch':         epoch,
            'model':         model.state_dict(),
            'optimizer':     optimizer.state_dict(),
            'scheduler':     scheduler.state_dict(),
            'best_val_loss': best_val_loss,
        }
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({**ckpt, 'best_val_loss': best_val_loss, 'val_psnr': val_psnr},
                       best_path)
            print(f"  → Best model saved  (val_l1={best_val_loss:.6f})")
        torch.save(ckpt, latest_path)

    wandb.finish()
    print(f"\nTraining complete.  Outputs: {args.out_dir}")


if __name__ == '__main__':
    main()
