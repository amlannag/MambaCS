#!/usr/bin/env python3
"""
inference.py — Run inference on a trained DcTNN experiment.

Usage:
    python inference.py --exp_dir ../Experiments/dctnn_baseline
    python inference.py --exp_dir ../Experiments/dctnn_baseline --num_images 5 --accel 4 --split val

Outputs a PDF with:
    Page 1  : Training metrics (loss, PSNR, LR vs epoch)
    Pages 2+: Per-image panels — GT, undersampled, reconstructed, k-spaces, MSE heatmap
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Ensure imports resolve regardless of where this script is called from
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from DcTNN.model import cascadeNet, axVIT, patchVIT
from DcTNN.dc import fft_2d, ifft_2d
from dataset import MRIDataset, load_mask


# ---------------------------------------------------------------------------
# Batch experiment list
# Each entry: {"exp_dir": "<path>", "accel": <int>, "num_images": <int>}
# ---------------------------------------------------------------------------

EXPERIMENTS = [
    {"exp_dir": "../Experiments/KSpace_patch",               "accel": 8, "num_images": 5},
    {"exp_dir": "../Experiments/KSpace_kaleidoscope",        "accel": 8, "num_images": 5},
    {"exp_dir": "../Experiments/KSpace_axial",               "accel": 8, "num_images": 5},
    {"exp_dir": "../Experiments/Lambda_Schedule_patch_cosine", "accel": 8, "num_images": 5},
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='DcTNN inference — generates a PDF report for a past experiment.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--exp_dir', type=str, default=None,
        help='Path to the experiment directory '
             '(must contain config.json, best_model.pth, metrics.json).',
    )
    parser.add_argument(
        '--all', action='store_true',
        help='Run residual reports for all experiments listed in EXPERIMENTS.',
    )
    parser.add_argument(
        '--num_images', type=int, default=5,
        help='Number of images to visualise.',
    )
    parser.add_argument(
        '--accel', type=int, default=4, choices=[4, 6, 8],
        help='Acceleration factor R for the undersampling mask.',
    )
    parser.add_argument(
        '--split', type=str, default='val', choices=['train', 'val'],
        help='Dataset split to draw images from.',
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Config normalisation
# ---------------------------------------------------------------------------

_DATA_KEYS  = {'data_dir', 'mask_dir', 'image_size', 'num_channels',
               'acceleration_factors', 'val_fraction', 'seed'}
_MODEL_KEYS = {'encoders', 'patch_size', 'nhead_patch', 'nhead_axial',
               'layer_no', 'num_encoder_layers', 'learned_lambda',
               'k_space_learning', 'lambda_schedule', 'lambda_start',
               'lambda_end', 'pos_emb_type', 'rope_theta', 'rope_mixed_rotate'}

def normalise_cfg(cfg_dict):
    """Accept both flat configs (saved by train.py) and nested configs."""
    if 'data' in cfg_dict and 'model' in cfg_dict:
        return cfg_dict
    return {
        'data':  {k: cfg_dict[k] for k in _DATA_KEYS  if k in cfg_dict},
        'model': {k: cfg_dict[k] for k in _MODEL_KEYS if k in cfg_dict},
    }


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

_ENCODER_MAP = {
    'axial':        lambda m, ch: (axVIT,    dict(layerNo=m['layer_no'], numCh=ch, d_model=None,
                                                   nhead=m['nhead_axial'],
                                                   num_encoder_layers=m['num_encoder_layers'],
                                                   dim_feedforward=None,
                                                   pos_emb_type=m.get('pos_emb_type', 'APE'),
                                                   rope_theta=m.get('rope_theta', 100.0),
                                                   rope_mixed_rotate=m.get('rope_mixed_rotate', True))),
    'kaleidoscope': lambda m, ch: (patchVIT, dict(patch_size=m['patch_size'], kaleidoscope=True,
                                                   layerNo=m['layer_no'], numCh=ch,
                                                   nhead=m['nhead_patch'],
                                                   num_encoder_layers=m['num_encoder_layers'],
                                                   dim_feedforward=None, d_model=None,
                                                   pos_emb_type=m.get('pos_emb_type', 'APE'),
                                                   rope_theta=m.get('rope_theta', 100.0),
                                                   rope_mixed_rotate=m.get('rope_mixed_rotate', True))),
    'patch':        lambda m, ch: (patchVIT, dict(patch_size=m['patch_size'], kaleidoscope=False,
                                                   layerNo=m['layer_no'], numCh=ch,
                                                   nhead=m['nhead_patch'],
                                                   num_encoder_layers=m['num_encoder_layers'],
                                                   dim_feedforward=None, d_model=None,
                                                   pos_emb_type=m.get('pos_emb_type', 'APE'),
                                                   rope_theta=m.get('rope_theta', 100.0),
                                                   rope_mixed_rotate=m.get('rope_mixed_rotate', True))),
}


def build_model_from_config(cfg_dict):
    m = cfg_dict['model']
    d = cfg_dict['data']

    ksl = m.get('k_space_learning', False)
    encoders = m.get('encoders', ['axial', 'kaleidoscope', 'patch'])
    k_space_list = list(ksl) if isinstance(ksl, (list, tuple)) else [bool(ksl)] * len(encoders)

    enc_list, enc_args = [], []
    for i, name in enumerate(encoders):
        # k-space stages receive 2-channel complex input; image stages use config num_channels
        ch = 2 if k_space_list[i] else d['num_channels']
        cls, args = _ENCODER_MAP[name](m, ch)
        enc_list.append(cls)
        enc_args.append(args)

    use_learned_lamb = m.get('lambda_schedule', 'none') == 'none'
    return cascadeNet(d['image_size'], enc_list, enc_args,
                      use_learned_lamb, k_space_learning=k_space_list)


# ---------------------------------------------------------------------------
# Undersampling (local copy so we do not depend on cfg global in train.py)
# ---------------------------------------------------------------------------

def simulate_undersampling(gt_batch, mask, num_channels=1, norm='ortho'):
    kspace_full = fft_2d(gt_batch)
    kspace_us   = kspace_full * mask
    zf_image    = ifft_2d(kspace_us, norm=norm)[:, 0:num_channels, :, :]
    return zf_image, kspace_us


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def to_image(tensor):
    """[B, C, H, W] → [H, W] magnitude numpy array."""
    return np.abs(tensor[0, 0].cpu().numpy())


def to_kspace_log(kspace_tensor):
    """[B, 2, H, W] k-space → log-magnitude [H, W], fft-shifted for display."""
    real = kspace_tensor[0, 0].cpu().numpy()
    imag = kspace_tensor[0, 1].cpu().numpy()
    mag  = np.sqrt(real ** 2 + imag ** 2)
    return np.log(np.fft.fftshift(mag) + 1e-8)


def psnr_numpy(pred, gt, max_val=1.0):
    mse = np.mean((pred - gt) ** 2)
    if mse == 0:
        return float('inf')
    return 20.0 * np.log10(max_val / np.sqrt(mse))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_metrics(metrics, pdf):
    epochs     = [m['epoch']      for m in metrics]
    train_loss = [m['train_loss'] for m in metrics]
    val_loss   = [m['val_loss']   for m in metrics]
    val_psnr   = [m['val_psnr']   for m in metrics]
    lr         = [m['lr']         for m in metrics]

    best_idx   = int(np.argmax(val_psnr))
    best_epoch = epochs[best_idx]
    best_psnr  = val_psnr[best_idx]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training Metrics', fontsize=16, fontweight='bold')

    # ---- Loss curves ----
    ax = axes[0, 0]
    ax.plot(epochs, train_loss, label='Train L1', color='tab:blue',   linewidth=2)
    ax.plot(epochs, val_loss,   label='Val L1',   color='tab:orange', linewidth=2)
    ax.axvline(best_epoch, color='red', linestyle='--', alpha=0.6,
               label=f'Best val epoch ({best_epoch})')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('L1 Loss')
    ax.set_title('Train / Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ---- Val PSNR ----
    ax = axes[0, 1]
    ax.plot(epochs, val_psnr, color='tab:green', linewidth=2)
    ax.axvline(best_epoch, color='red', linestyle='--', alpha=0.6,
               label=f'Best: {best_psnr:.2f} dB @ epoch {best_epoch}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('PSNR (dB)')
    ax.set_title('Validation PSNR')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ---- Learning rate ----
    ax = axes[1, 0]
    ax.plot(epochs, lr, color='tab:purple', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # ---- Train vs Val (overfitting check) ----
    ax = axes[1, 1]
    sc = ax.scatter(train_loss, val_loss, c=epochs, cmap='viridis',
                    s=20, alpha=0.8)
    plt.colorbar(sc, ax=ax, label='Epoch')
    ax.set_xlabel('Train L1 Loss')
    ax.set_ylabel('Val L1 Loss')
    ax.set_title('Train vs Val Loss')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def plot_image_results(gt, zf_image, recon, kspace_gt, kspace_us, kspace_recon,
                       img_idx, accel, pdf):
    gt_np    = to_image(gt)
    zf_np    = to_image(zf_image)
    recon_np = to_image(recon)

    ks_gt_np    = to_kspace_log(kspace_gt)
    ks_us_np    = to_kspace_log(kspace_us)
    ks_recon_np = to_kspace_log(kspace_recon)

    mse_map  = (recon_np - gt_np) ** 2
    mse_val  = float(np.mean(mse_map))
    psnr_val = psnr_numpy(recon_np, gt_np)

    fig, axes = plt.subplots(2, 4, figsize=(22, 11))
    fig.suptitle(f'Image {img_idx + 1}  —  Acceleration R={accel}  |  '
                 f'PSNR = {psnr_val:.2f} dB  |  MSE = {mse_val:.2e}',
                 fontsize=13, fontweight='bold')

    def show(ax, data, title, cmap='gray', add_cbar=False):
        im = ax.imshow(data, cmap=cmap, origin='upper')
        ax.set_title(title, fontsize=10)
        ax.axis('off')
        if add_cbar:
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        return im

    # Top row: spatial domain
    show(axes[0, 0], gt_np,    'Ground Truth')
    show(axes[0, 1], zf_np,    f'Undersampled  (R={accel})')
    show(axes[0, 2], recon_np, 'Reconstructed')
    show(axes[0, 3], mse_map,  'MSE Heatmap\n(Recon − GT)²', cmap='hot', add_cbar=True)

    # Bottom row: k-space domain
    show(axes[1, 0], ks_gt_np,    'GT K-space  (log|·|)',             cmap='inferno')
    show(axes[1, 1], ks_us_np,    f'Undersampled K-space  (log|·|)',  cmap='inferno')
    show(axes[1, 2], ks_recon_np, 'Reconstructed K-space  (log|·|)', cmap='inferno')

    # Metrics panel
    axes[1, 3].axis('off')
    axes[1, 3].text(
        0.08, 0.55,
        f"Per-Image Metrics\n"
        f"{'─' * 24}\n"
        f"PSNR  : {psnr_val:>8.3f} dB\n"
        f"MSE   : {mse_val:>8.2e}\n"
        f"MaxErr: {float(np.max(mse_map)):>8.2e}",
        transform=axes[1, 3].transAxes,
        fontsize=12, verticalalignment='center',
        fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85),
    )

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary page
# ---------------------------------------------------------------------------

def plot_summary(results, accel, exp_dir, pdf):
    """Aggregate metrics table across all images."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')

    headers = ['Image', 'PSNR (dB)', 'MSE']
    rows = [[f'Image {i+1}', f'{r["psnr"]:.3f}', f'{r["mse"]:.2e}']
            for i, r in enumerate(results)]
    avg_psnr = np.mean([r['psnr'] for r in results])
    avg_mse  = np.mean([r['mse']  for r in results])
    rows.append(['Mean', f'{avg_psnr:.3f}', f'{avg_mse:.2e}'])

    table = ax.table(
        cellText=rows, colLabels=headers,
        loc='center', cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.4, 2.0)

    # Highlight mean row
    for col in range(len(headers)):
        table[len(rows), col].set_facecolor('#d0e8ff')

    ax.set_title(
        f'Summary — {os.path.basename(exp_dir)}  (R={accel})',
        fontsize=14, fontweight='bold', pad=20,
    )
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Residual report
# ---------------------------------------------------------------------------

def plot_residual_page(gt, zf_image, recon, kspace_us, kspace_recon, img_idx, accel, pdf):
    """2×3 panel: undersampled image/kspace, reconstructed image/kspace, residual map + histogram."""
    gt_np    = to_image(gt)
    zf_np    = to_image(zf_image)
    recon_np = to_image(recon)
    residual = recon_np - gt_np

    ks_us_np    = to_kspace_log(kspace_us)
    ks_recon_np = to_kspace_log(kspace_recon)

    psnr_val = psnr_numpy(recon_np, gt_np)
    mse_val  = float(np.mean(residual ** 2))

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(
        f'Image {img_idx + 1}  —  R={accel}  |  PSNR = {psnr_val:.2f} dB  |  MSE = {mse_val:.2e}',
        fontsize=13, fontweight='bold',
    )

    vmax = gt_np.max()

    # Row 0: spatial
    axes[0, 0].imshow(zf_np,    cmap='gray', vmin=0, vmax=vmax, origin='upper')
    axes[0, 0].set_title(f'Undersampled Image  (R={accel})', fontsize=10)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(recon_np, cmap='gray', vmin=0, vmax=vmax, origin='upper')
    axes[0, 1].set_title('Reconstructed Image', fontsize=10)
    axes[0, 1].axis('off')

    abs_max = np.max(np.abs(residual))
    im_res = axes[0, 2].imshow(residual, cmap='RdBu_r', vmin=-abs_max, vmax=abs_max, origin='upper')
    axes[0, 2].set_title('Residual Map  (Recon − GT)', fontsize=10)
    axes[0, 2].axis('off')
    plt.colorbar(im_res, ax=axes[0, 2], fraction=0.046, pad=0.04)

    # Row 1: k-space + histogram
    axes[1, 0].imshow(ks_us_np,    cmap='inferno', origin='upper')
    axes[1, 0].set_title('Undersampled K-space  (log|·|)', fontsize=10)
    axes[1, 0].axis('off')

    axes[1, 1].imshow(ks_recon_np, cmap='inferno', origin='upper')
    axes[1, 1].set_title('Reconstructed K-space  (log|·|)', fontsize=10)
    axes[1, 1].axis('off')

    axes[1, 2].hist(residual.ravel(), bins=80, color='steelblue', edgecolor='none', alpha=0.85)
    axes[1, 2].axvline(0, color='red', linewidth=1.5, linestyle='--')
    axes[1, 2].set_xlabel('Residual value')
    axes[1, 2].set_ylabel('Count')
    axes[1, 2].set_title('Residual Distribution', fontsize=10)
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def generate_residual_report(exp_dir, accel=8, num_images=5, split='val'):
    """Generate residual-focused PDF for one experiment and save it inside exp_dir."""
    exp_dir = os.path.abspath(exp_dir)
    if not os.path.isdir(exp_dir):
        print(f"  [SKIP] Directory not found: {exp_dir}")
        return

    print(f"\n{'='*60}")
    print(f"Residual report: {exp_dir}")
    print(f"{'='*60}")

    config_path = os.path.join(exp_dir, 'config.json')
    best_path   = os.path.join(exp_dir, 'best_model.pth')
    for p in (config_path, best_path):
        if not os.path.exists(p):
            print(f"  [SKIP] Missing file: {p}")
            return

    with open(config_path) as f:
        cfg_dict = normalise_cfg(json.load(f))

    N            = cfg_dict['data']['image_size']
    num_channels = cfg_dict['data']['num_channels']
    device       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = build_model_from_config(cfg_dict).to(device)
    ckpt  = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    mask_path  = os.path.join(script_dir, cfg_dict['data']['mask_dir'], f'mask_R{accel}.png')
    if not os.path.exists(mask_path):
        print(f"  [SKIP] Mask not found: {mask_path}")
        return
    mask = load_mask(mask_path, N).to(device)

    data_dir     = cfg_dict['data']['data_dir']
    val_fraction = cfg_dict['data'].get('val_fraction', 0.1)
    seed         = cfg_dict['data'].get('seed', 42)
    dataset      = MRIDataset(data_dir, N=N, split=split,
                              val_fraction=val_fraction, seed=seed)
    num_images   = min(num_images, len(dataset))
    indices      = np.linspace(0, len(dataset) - 1, num_images, dtype=int)

    pdf_path = os.path.join(exp_dir, 'residual_report.pdf')
    with PdfPages(pdf_path) as pdf:
        d           = pdf.infodict()
        d['Title']  = f'Residual Report — {os.path.basename(exp_dir)}'
        d['Author'] = 'inference.py'

        for page_idx, dataset_idx in enumerate(indices):
            print(f"  Image {page_idx + 1}/{num_images}")
            gt = dataset[int(dataset_idx)].unsqueeze(0).to(device)

            with torch.no_grad():
                zf_image, kspace_us = simulate_undersampling(gt, mask, num_channels)
                recon               = model(zf_image, kspace_us, mask)
                kspace_recon        = fft_2d(recon)

            plot_residual_page(
                gt.cpu(), zf_image.cpu(), recon.cpu(),
                kspace_us.cpu(), kspace_recon.cpu(),
                page_idx, accel, pdf,
            )

    print(f"  Saved: {pdf_path}")


def run_all_experiments(split='val'):
    """Run generate_residual_report for every entry in EXPERIMENTS."""
    for entry in EXPERIMENTS:
        generate_residual_report(
            exp_dir    = entry['exp_dir'],
            accel      = entry.get('accel', 8),
            num_images = entry.get('num_images', 5),
            split      = split,
        )
    print("\nAll done.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_inference(exp_dir, num_images=5, accel=4, split='val'):
    exp_dir = os.path.abspath(exp_dir)

    if not os.path.isdir(exp_dir):
        raise FileNotFoundError(f"Experiment directory does not exist: {exp_dir}")

    print(f"\n{'='*60}")
    print(f"Running inference on: {exp_dir}")
    print(f"{'='*60}")

    # ---- Config ----
    config_path = os.path.join(exp_dir, 'config.json')
    with open(config_path) as f:
        cfg_dict = normalise_cfg(json.load(f))

    N            = cfg_dict['data']['image_size']
    num_channels = cfg_dict['data']['num_channels']
    print(f"Config         : image_size={N}, num_channels={num_channels}")

    # ---- Metrics ----
    metrics_path = os.path.join(exp_dir, 'metrics.json')
    with open(metrics_path) as f:
        metrics = json.load(f)
    print(f"Metrics        : {len(metrics)} epochs logged")

    # ---- Device ----
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device         : {device}")

    # ---- Model ----
    best_path = os.path.join(exp_dir, 'best_model.pth')
    model = build_model_from_config(cfg_dict).to(device)
    ckpt  = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    best_epoch    = ckpt.get('epoch', '?')
    best_val_loss = ckpt.get('best_val_loss', float('nan'))
    best_val_psnr = ckpt.get('val_psnr',      float('nan'))
    print(f"Checkpoint     : best_model.pth  "
          f"(epoch={best_epoch}, val_loss={best_val_loss:.4f}, "
          f"val_psnr={best_val_psnr:.2f} dB)")

    # ---- Mask ----
    script_dir = os.path.dirname(os.path.abspath(__file__))
    mask_dir   = os.path.join(script_dir, cfg_dict['data']['mask_dir'])
    mask_path  = os.path.join(mask_dir, f'mask_R{accel}.png')
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask not found: {mask_path}")
    mask = load_mask(mask_path, N).to(device)
    print(f"Mask           : R={accel}, shape={tuple(mask.shape)}")

    # ---- Dataset ----
    data_dir     = cfg_dict['data']['data_dir']
    val_fraction = cfg_dict['data'].get('val_fraction', 0.1)
    seed         = cfg_dict['data'].get('seed', 42)

    dataset    = MRIDataset(data_dir, N=N, split=split,
                            val_fraction=val_fraction, seed=seed)
    num_images = min(num_images, len(dataset))
    indices    = np.linspace(0, len(dataset) - 1, num_images, dtype=int)
    print(f"Dataset        : {len(dataset)} images in '{split}' split")
    print(f"Visualising    : {num_images} images\n")

    # ---- Generate PDF ----
    pdf_path = os.path.join(exp_dir, 'inference_results.pdf')
    results  = []

    with PdfPages(pdf_path) as pdf:

        print("Plotting training metrics...")
        plot_metrics(metrics, pdf)

        for page_idx, dataset_idx in enumerate(indices):
            print(f"  Image {page_idx + 1}/{num_images}  (dataset index {int(dataset_idx)})")

            gt = dataset[int(dataset_idx)].unsqueeze(0).to(device)

            with torch.no_grad():
                zf_image, kspace_us = simulate_undersampling(gt, mask, num_channels)
                recon               = model(zf_image, kspace_us, mask)
                kspace_gt           = fft_2d(gt)
                kspace_recon        = fft_2d(recon)

            gt_np    = to_image(gt.cpu())
            recon_np = to_image(recon.cpu())
            mse_val  = float(np.mean((recon_np - gt_np) ** 2))
            psnr_val = psnr_numpy(recon_np, gt_np)
            results.append({'psnr': psnr_val, 'mse': mse_val})

            plot_image_results(
                gt.cpu(), zf_image.cpu(), recon.cpu(),
                kspace_gt.cpu(), kspace_us.cpu(), kspace_recon.cpu(),
                page_idx, accel, pdf,
            )

        plot_summary(results, accel, exp_dir, pdf)

        d           = pdf.infodict()
        d['Title']  = f'DcTNN Inference — {os.path.basename(exp_dir)}'
        d['Author'] = 'inference.py'

    print(f"\nPDF saved to: {pdf_path}")
    print("\n--- Summary ---")
    for i, r in enumerate(results):
        print(f"  Image {i+1}: PSNR={r['psnr']:.3f} dB  MSE={r['mse']:.2e}")
    print(f"  Mean  : PSNR={np.mean([r['psnr'] for r in results]):.3f} dB  "
          f"MSE={np.mean([r['mse'] for r in results]):.2e}")


def main():
    args = parse_args()
    if args.all:
        run_all_experiments(split=args.split)
    else:
        run_inference(args.exp_dir, args.num_images, args.accel, args.split)


if __name__ == '__main__':
    main()
