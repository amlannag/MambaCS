"""
notebooks/notebook_inference.py
Shared inference utilities for all MambaCS notebooks.
"""

import os
import sys
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from glob import glob
from PIL import Image

# Ensure MambaCS root is importable when loaded from the notebooks/ directory
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import h5py
from skimage.restoration import denoise_tv_chambolle
from fastmri.data.subsample import RandomMaskFunc

from DcTNN.dc import fft_2d, ifft_2d
from train_utils import simulate_undersampling, build_model, FastMRIMaskGenerator
from config import Config


# ── Metrics ───────────────────────────────────────────────────────────────────

def psnr(gt_np, pred_np):
    mse = np.mean((gt_np - pred_np) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(gt_np.max() / (np.sqrt(mse) + 1e-12))


def ssim(gt, pred):
    """Luminance-only SSIM proxy."""
    mu_g, mu_p = gt.mean(), pred.mean()
    c1 = (0.01 * gt.max()) ** 2
    c2 = (0.03 * gt.max()) ** 2
    sig_gp = np.mean((gt - mu_g) * (pred - mu_p))
    return ((2 * mu_g * mu_p + c1) * (2 * sig_gp + c2)) / \
           ((mu_g ** 2 + mu_p ** 2 + c1) * (np.var(gt) + np.var(pred) + c2))


# ── Image / k-space utilities ─────────────────────────────────────────────────

def to_magnitude(t):
    """[B,1,H,W] complex or real tensor → [H,W] float numpy."""
    return torch.abs(t)[0, 0].cpu().float().numpy()


def ks_log_mag(ks_tensor):
    """[B,1,H,W] complex k-space tensor → log magnitude [H,W] numpy array."""
    return np.log1p(torch.abs(ks_tensor)[0, 0].cpu().numpy())


def img_to_log_kspace(img_np, device):
    """[H,W] float numpy image → log |FFT| [H,W] numpy array."""
    t = torch.tensor(img_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(torch.complex64).to(device)
    return np.log1p(torch.abs(fft_2d(t))[0, 0].cpu().numpy())


def png_to_kspace(img_np, image_size, device):
    """
    Resize a [H,W] float32 image and return centered k-space [1,1,H,W] complex64.
    FFT convention matches OASISDataset / fft_2d in DcTNN/dc.py.
    """
    H, W = image_size
    img_u8 = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
    img_t = torch.tensor(
        np.array(Image.fromarray(img_u8).resize((W, H), Image.LANCZOS), dtype='float32') / 255.0
    )
    kspace = torch.fft.fftshift(
        torch.fft.fft2(torch.fft.ifftshift(img_t), norm='ortho')
    )
    return kspace.unsqueeze(0).unsqueeze(0).to(torch.complex64).to(device)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_test_images(data_dir):
    """Load all PNG slices from a directory. Returns list of (filename, [H,W] float32 in [0,1])."""
    paths = sorted(glob(os.path.join(data_dir, '*.png')))
    if not paths:
        raise ValueError(
            f'No PNG files found in {data_dir!r}.\n'
            f'Drop your test PNG slices into that folder and re-run.'
        )
    images = [
        (os.path.basename(p), np.array(Image.open(p).convert('L'), dtype=np.float32) / 255.0)
        for p in paths
    ]
    print(f'Loaded {len(images)} PNG slices from {data_dir!r}')
    return images


def load_all_slices(h5_file, kspace_key='kspace', image_size=(320, 320)):
    """
    Load every k-space slice from a .h5 file, center-cropped to image_size.
    Returns list of [1,1,H,W] complex64 tensors.
    """
    from dataset import center_crop_kspace
    with h5py.File(h5_file, 'r') as f:
        if kspace_key not in f:
            raise KeyError(f'Key "{kspace_key}" not found — available: {list(f.keys())}')
        ks_vol = f[kspace_key][:]
    slices = []
    for sl in ks_vol:
        ks = center_crop_kspace(sl, image_size)
        slices.append(torch.tensor(ks, dtype=torch.complex64).unsqueeze(0).unsqueeze(0))
    print(f'Loaded {len(slices)} slices from {os.path.basename(h5_file)}'
          f' — shape {slices[0].shape}  dtype {slices[0].dtype}')
    return slices


# ── Mask utilities ────────────────────────────────────────────────────────────

_DEFAULT_CENTER_FRACTIONS = {4: 0.08, 6: 0.04, 8: 0.04}


def default_center_fraction(accel, center_fractions=None):
    cf = center_fractions or _DEFAULT_CENTER_FRACTIONS
    return cf.get(int(accel), 0.04 if int(accel) >= 6 else 0.08)


def generate_column_mask(image_size, accel, device, center_fraction=None, seed=None):
    """Central ACS + random outer columns undersampling mask."""
    H, W = image_size if isinstance(image_size, (tuple, list)) else (image_size, image_size)
    cf = default_center_fraction(accel) if center_fraction is None else center_fraction
    num_center = max(1, int(round(W * cf)))
    num_total = max(num_center, W // int(accel))

    mask_1d = torch.zeros(W, dtype=torch.float32, device=device)
    c0 = (W - num_center) // 2
    mask_1d[c0:c0 + num_center] = 1.0

    num_outer = num_total - num_center
    if num_outer > 0:
        outer = (mask_1d == 0).nonzero(as_tuple=True)[0]
        g = torch.Generator(device=device)
        if seed is not None:
            g.manual_seed(hash((int(accel), int(seed))) & 0xFFFFFFFF)
        perm = torch.randperm(len(outer), device=device, generator=g)
        mask_1d[outer[perm[:num_outer]]] = 1.0

    return mask_1d.unsqueeze(0).repeat(H, 1)


def generate_fastmri_random_mask(image_size, accel, device, seed=1234, center_fraction=None):
    """FastMRI-style random mask. Returns (mask [H,W], num_low_frequencies)."""
    H, W = image_size if isinstance(image_size, (tuple, list)) else (image_size, image_size)
    cf = default_center_fraction(accel) if center_fraction is None else center_fraction
    mask_func = RandomMaskFunc(center_fractions=[cf], accelerations=[accel], seed=seed)
    mask, num_low = mask_func((1, H, W, 1), seed=seed)
    mask_1d = mask.squeeze().to(torch.float32)
    if mask_1d.ndim != 1:
        mask_1d = mask_1d.reshape(-1)
    return mask_1d.unsqueeze(0).repeat(H, 1).to(device), int(num_low)


def get_mask(mask_source, image_size, accel, device, seed=None):
    if mask_source == 'current':
        return generate_column_mask(image_size, accel, device, seed=seed)
    if mask_source == 'fastmri_random':
        mask, _ = generate_fastmri_random_mask(image_size, accel, device, seed=seed or 1234)
        return mask
    raise ValueError(f'Unknown mask_source: {mask_source!r}. Choose from: current, fastmri_random')


def describe_mask(mask):
    mask_np = mask.detach().cpu().numpy()
    return {
        'sampled_fraction': float(mask_np.mean()),
        'sampled_columns': int(mask_np[mask_np.shape[0] // 2].sum()),
    }


def zero_filled_stats(kspace_full_tensor, mask, device):
    ks = kspace_full_tensor.to(device)
    gt_np = torch.abs(ifft_2d(ks))[0, 0].cpu().numpy()
    zf_np = torch.abs(ifft_2d(ks * mask))[0, 0].cpu().numpy()
    stats = describe_mask(mask)
    stats.update({'gt_np': gt_np, 'zf_np': zf_np,
                  'psnr': psnr(gt_np, zf_np), 'ssim': ssim(gt_np, zf_np)})
    return stats


def mask_summary_table(image_size, device, accels=(4, 6, 8)):
    rows = []
    for accel in accels:
        current_mask = generate_column_mask(image_size, accel, device)
        fastmri_mask, low = generate_fastmri_random_mask(image_size, accel, device)
        c = describe_mask(current_mask)
        f = describe_mask(fastmri_mask)
        rows.append({'R': accel, 'current_frac': c['sampled_fraction'],
                     'current_cols': c['sampled_columns'], 'fastmri_frac': f['sampled_fraction'],
                     'fastmri_cols': f['sampled_columns'], 'fastmri_low': low})
    print(f"{'R':<4} {'current frac':>14} {'current cols':>14} {'fastmri frac':>14} {'fastmri cols':>14} {'fastmri low':>14}")
    print('-' * 82)
    for row in rows:
        print(f"{row['R']:<4} {row['current_frac']:>14.4f} {row['current_cols']:>14d} "
              f"{row['fastmri_frac']:>14.4f} {row['fastmri_cols']:>14d} {row['fastmri_low']:>14d}")
    return rows


def verify_fastmri_mask(H, W, accel, seed=1234, center_fractions=None):
    """Diagnostic: compare first-principles vs fastMRI repo mask implementation."""
    cf = default_center_fraction(accel, center_fractions)
    fp_mask, fp_low = _fastmri_mask_np(H, W, accel, cf, seed=seed)
    repo_func = RandomMaskFunc(center_fractions=[cf], accelerations=[accel], seed=seed)
    repo_raw, repo_low = repo_func((1, H, W, 1), seed=seed)
    repo_mask = np.repeat(repo_raw.squeeze().numpy().astype(np.float32)[None, :], H, axis=0)
    diff = np.abs(fp_mask - repo_mask)
    print(f'  first-principles  mean={fp_mask.mean():.4f}  low={fp_low}')
    print(f'  repo helper       mean={repo_mask.mean():.4f}  low={int(repo_low)}')
    print(f'  differing entries: {int(diff.sum())}')
    return fp_mask, repo_mask


# ── Config & model loading ────────────────────────────────────────────────────

def _flat_dict_to_cfg(flat):
    """Reconstruct a Config object from the flat dict saved by train.py."""
    cfg = Config()
    for k, v in flat.items():
        if not hasattr(cfg, k):
            continue
        if k in ('image_size', 'patch_size') and isinstance(v, list):
            v = tuple(v)
        setattr(cfg, k, v)
    return cfg


def load_experiment(label, exp_dir, device):
    """Load a trained experiment from disk. Returns (model, cfg)."""
    config_path = os.path.join(exp_dir, 'config.json')
    ckpt_path = os.path.join(exp_dir, 'best_model.pth')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f'config.json not found in {exp_dir}')
    with open(config_path) as f:
        cfg = _flat_dict_to_cfg(json.load(f))
    model = build_model(cfg).to(device)
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt['model'])
        ep = ckpt.get('epoch', '?')
        p = ckpt.get('val_psnr', float('nan'))
        print(f'  [{label}]  epoch={ep}  val_psnr={p:.2f} dB  ({exp_dir})')
    else:
        print(f'  [{label}]  WARNING — no checkpoint, using random weights')
    model.eval()
    return model, cfg


def load_experiment_set(experiment_dirs, device):
    """Load multiple experiments. Returns (models_dict, cfgs_dict)."""
    print('Loading models...')
    models, cfgs = {}, {}
    for label, exp_dir in experiment_dirs.items():
        models[label], cfgs[label] = load_experiment(label, exp_dir, device)
    print('Done.')
    return models, cfgs


# ── Inference ─────────────────────────────────────────────────────────────────

def denormalize_recon(recon, stats, learning):
    """
    Invert zscore normalisation → [H,W] float32 numpy magnitude image in original pixel domain.
    For k_space mode: IFFT output then invert real/imag separately.
    """
    if not stats or 'mean_r' not in stats:
        if recon.is_complex():
            return torch.abs(ifft_2d(recon))[0, 0].cpu().numpy()
        return recon[0, 0].cpu().numpy()
    mean_r, std_r = stats['mean_r'], stats['std_r']
    mean_i, std_i = stats['mean_i'], stats['std_i']
    if learning == 'k_space':
        img_norm = ifft_2d(recon)
        return torch.abs(torch.complex(
            img_norm.real * std_r + mean_r,
            img_norm.imag * std_i + mean_i,
        ))[0, 0].cpu().numpy()
    return (recon * std_r + mean_r)[0, 0].cpu().numpy()


def run_inference_kspace(kspace_tensor, models, cfgs, image_size, accel, device,
                         mask_source='current', mask_seed=None):
    """
    Run all models on a single [1,1,H,W] complex k-space tensor.
    Returns (results_dict, gt_np, zf_np, mask).
    results_dict: label → {recon [H,W], psnr, ssim}
    """
    mask = get_mask(mask_source, image_size, accel, device, seed=mask_seed)
    ks = kspace_tensor.to(device)
    gt_np = torch.abs(ifft_2d(ks))[0, 0].cpu().numpy()
    zf_np = torch.abs(ifft_2d(ks * mask))[0, 0].cpu().numpy()

    results = {}
    for label, model in models.items():
        learning = cfgs[label].learning
        norm = cfgs[label].norm or 'none'
        model_input, dc_input, _, stats = simulate_undersampling(
            ks, mask, learning=learning, norm=norm
        )
        with torch.no_grad():
            recon = model(model_input, dc_input, mask)
        recon_mag = denormalize_recon(recon, stats, learning)
        results[label] = {
            'recon': recon_mag,
            'psnr': psnr(gt_np, recon_mag),
            'ssim': ssim(gt_np, recon_mag),
        }
    return results, gt_np, zf_np, mask


def run_inference_png(img_np, experiments, device, seed=42, tv_weight=0.1):
    """
    Run all OASIS experiments + TV baseline on one [H,W] float image in [0,1].
    Returns results dict: name → {recon, gt, zf, accel, image_size, psnr, ssim, zf_psnr, zf_ssim}
    """
    from inference import _denormalize_image
    results = {}
    for name, exp_data in experiments.items():
        cfg = exp_data['config']
        model = exp_data['model']
        data_cfg = cfg['data']
        model_cfg = cfg['model']

        image_size = tuple(int(x) for x in data_cfg['image_size'])
        learning = model_cfg.get('learning', 'image')
        norm = data_cfg.get('norm', None)
        accel = int(data_cfg.get('acceleration_factors', [4])[0])

        kspace_full = png_to_kspace(img_np, image_size, device)
        mask_gen = FastMRIMaskGenerator([accel])
        kspace_us, mask, _ = mask_gen.apply(kspace_full, accel, seed=seed)

        model_input, dc_input, _gt_norm, stats = simulate_undersampling(
            kspace_full, mask, learning=learning, norm=norm, kspace_us=kspace_us
        )
        with torch.no_grad():
            recon_norm = model(model_input, dc_input, mask)

        if learning == 'k_space':
            recon_img = _denormalize_image(ifft_2d(recon_norm), stats)
        else:
            recon_img = _denormalize_image(recon_norm, stats)
        recon_np = to_magnitude(recon_img)

        H, W = image_size
        img_u8 = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
        gt_np = np.array(Image.fromarray(img_u8).resize((W, H), Image.LANCZOS), dtype='float32') / 255.0
        zf_np = to_magnitude(ifft_2d(kspace_us))

        results[name] = dict(
            recon=recon_np, gt=gt_np, zf=zf_np, accel=accel, image_size=image_size,
            psnr=psnr(gt_np, recon_np), ssim=ssim(gt_np, recon_np),
            zf_psnr=psnr(gt_np, zf_np), zf_ssim=ssim(gt_np, zf_np),
        )

    if results:
        ref = next(iter(results.values()))
        tv_recon = denoise_tv_chambolle(ref['zf'], weight=tv_weight)
        results[f'TV (w={tv_weight})'] = dict(
            recon=tv_recon, gt=ref['gt'], zf=ref['zf'],
            accel=ref['accel'], image_size=ref['image_size'],
            psnr=psnr(ref['gt'], tv_recon), ssim=ssim(ref['gt'], tv_recon),
            zf_psnr=ref['zf_psnr'], zf_ssim=ref['zf_ssim'],
        )
    return results


# ── Plot helpers ──────────────────────────────────────────────────────────────

def _make_grid_figure(n_rows, col_headers=None):
    fig, axes = plt.subplots(n_rows, 5, figsize=(20, 4 * n_rows),
                             gridspec_kw={'width_ratios': [0.7, 3, 3, 3, 2]})
    if col_headers:
        for col, text in enumerate(col_headers):
            axes[0, col].set_title(text, fontsize=9, fontweight='bold')
    return fig, axes


def _set_row_label(axes, row, text):
    ax = axes[row, 0]
    ax.axis('off')
    ax.text(0.5, 0.5, text, ha='center', va='center', fontsize=9, fontweight='bold',
            transform=ax.transAxes, multialignment='center')


def _fill_metrics_cell(ax, p, s):
    ax.text(0.5, 0.60, f'PSNR\n{p:.2f} dB', ha='center', va='center',
            fontsize=14, fontweight='bold', transform=ax.transAxes)
    ax.text(0.5, 0.30, f'SSIM\n{s:.4f}', ha='center', va='center',
            fontsize=12, transform=ax.transAxes)
    ax.axis('off')


_COL_HEADERS = ['', 'Image', 'K-space  log|·|', 'Error map', 'PSNR / SSIM']


def plot_inference_kspace(kspace_tensor, models, cfgs, image_size, accel, device,
                          title='Knee MRI', mask_source='current', mask_seed=None):
    results, gt_np, zf_np, mask = run_inference_kspace(
        kspace_tensor, models, cfgs, image_size, accel, device,
        mask_source=mask_source, mask_seed=mask_seed,
    )
    labels = list(results.keys())

    ks = kspace_tensor.to(device)
    gt_ks = np.log1p(torch.abs(ks)[0, 0].cpu().numpy())
    zf_ks = np.log1p(torch.abs(ks * mask)[0, 0].cpu().numpy())
    recon_ks_maps = [img_to_log_kspace(r['recon'], device) for r in results.values()]
    err_maps = [np.abs(gt_np - r['recon']) for r in results.values()]
    zf_err = np.abs(gt_np - zf_np)

    ks_max = np.percentile(np.concatenate([k.ravel() for k in [gt_ks, zf_ks] + recon_ks_maps]), 99)
    err_max = np.percentile(np.concatenate([e.ravel() for e in err_maps + [zf_err]]), 99)
    vmax = np.percentile(gt_np, 99)

    fig, axes = _make_grid_figure(2 + len(labels), _COL_HEADERS)
    fig.suptitle(f'{title}  --  R={accel}  {mask_source.replace("_", " ")} undersampling',
                 fontsize=13, fontweight='bold', y=1.01)

    _set_row_label(axes, 0, 'Ground\nTruth')
    axes[0, 1].imshow(gt_np, cmap='gray', vmin=0, vmax=vmax); axes[0, 1].axis('off')
    axes[0, 2].imshow(gt_ks, cmap='inferno', vmin=0, vmax=ks_max); axes[0, 2].axis('off')
    axes[0, 3].axis('off'); axes[0, 4].axis('off')

    _set_row_label(axes, 1, f'Zero-filled\nR={accel}')
    axes[1, 1].imshow(zf_np, cmap='gray', vmin=0, vmax=vmax); axes[1, 1].axis('off')
    axes[1, 2].imshow(zf_ks, cmap='inferno', vmin=0, vmax=ks_max); axes[1, 2].axis('off')
    axes[1, 3].imshow(zf_err, cmap='hot', vmin=0, vmax=err_max); axes[1, 3].axis('off')
    _fill_metrics_cell(axes[1, 4], psnr(gt_np, zf_np), ssim(gt_np, zf_np))

    for row, (label, ks_arr, err_arr) in enumerate(zip(labels, recon_ks_maps, err_maps), start=2):
        r = results[label]
        _set_row_label(axes, row, label)
        axes[row, 1].imshow(r['recon'], cmap='gray', vmin=0, vmax=vmax); axes[row, 1].axis('off')
        axes[row, 2].imshow(ks_arr, cmap='inferno', vmin=0, vmax=ks_max); axes[row, 2].axis('off')
        axes[row, 3].imshow(err_arr, cmap='hot', vmin=0, vmax=err_max); axes[row, 3].axis('off')
        _fill_metrics_cell(axes[row, 4], r['psnr'], r['ssim'])

    plt.tight_layout()
    plt.show()
    plt.close(fig)


def plot_inference_png(img_np, experiments, device, title='OASIS Brain MRI', seed=42, tv_weight=0.1):
    results = run_inference_png(img_np, experiments, device, seed=seed, tv_weight=tv_weight)
    if not results:
        print('No experiments loaded — nothing to plot.')
        return

    names = list(results.keys())
    ref = results[names[0]]
    gt_np, zf_np, accel = ref['gt'], ref['zf'], ref['accel']

    gt_ks = img_to_log_kspace(gt_np, device)
    zf_ks = img_to_log_kspace(zf_np, device)
    recon_ks_maps = [img_to_log_kspace(r['recon'], device) for r in results.values()]
    err_maps = [np.abs(r['gt'] - r['recon']) for r in results.values()]
    zf_err = np.abs(gt_np - zf_np)

    ks_max = np.percentile(np.concatenate([k.ravel() for k in [gt_ks, zf_ks] + recon_ks_maps]), 99)
    err_max = np.percentile(np.concatenate([e.ravel() for e in err_maps + [zf_err]]), 99)
    vmax = np.percentile(gt_np, 99)

    fig, axes = _make_grid_figure(2 + len(names), _COL_HEADERS)
    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.01)

    _set_row_label(axes, 0, 'Ground\nTruth')
    axes[0, 1].imshow(gt_np, cmap='gray', vmin=0, vmax=vmax); axes[0, 1].axis('off')
    axes[0, 2].imshow(gt_ks, cmap='inferno', vmin=0, vmax=ks_max); axes[0, 2].axis('off')
    axes[0, 3].axis('off'); axes[0, 4].axis('off')

    _set_row_label(axes, 1, f'Zero-filled\nR={accel}')
    axes[1, 1].imshow(zf_np, cmap='gray', vmin=0, vmax=vmax); axes[1, 1].axis('off')
    axes[1, 2].imshow(zf_ks, cmap='inferno', vmin=0, vmax=ks_max); axes[1, 2].axis('off')
    axes[1, 3].imshow(zf_err, cmap='hot', vmin=0, vmax=err_max); axes[1, 3].axis('off')
    _fill_metrics_cell(axes[1, 4], ref['zf_psnr'], ref['zf_ssim'])

    for row, (name, ks_arr, err_arr) in enumerate(zip(names, recon_ks_maps, err_maps), start=2):
        r = results[name]
        _set_row_label(axes, row, f'{name}\nR={r["accel"]}')
        axes[row, 1].imshow(r['recon'], cmap='gray', vmin=0, vmax=vmax); axes[row, 1].axis('off')
        axes[row, 2].imshow(ks_arr, cmap='inferno', vmin=0, vmax=ks_max); axes[row, 2].axis('off')
        axes[row, 3].imshow(err_arr, cmap='hot', vmin=0, vmax=err_max); axes[row, 3].axis('off')
        _fill_metrics_cell(axes[row, 4], r['psnr'], r['ssim'])

    plt.tight_layout()
    plt.show()
    plt.close(fig)


# ── TV baseline ───────────────────────────────────────────────────────────────

def fft2c_np(img):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img, axes=(-2, -1)), norm='ortho'), axes=(-2, -1))


def ifft2c_np(ks):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(ks, axes=(-2, -1)), norm='ortho'), axes=(-2, -1))


def _center_crop_np(arr, out_shape):
    h, w = arr.shape
    out_h, out_w = out_shape
    return arr[(h - out_h) // 2:(h - out_h) // 2 + out_h,
               (w - out_w) // 2:(w - out_w) // 2 + out_w]


def _fastmri_mask_np(H, W, accel, center_fraction, seed=1234):
    num_low = max(1, min(int(round(W * center_fraction)), W))
    prob = float(np.clip((W / accel - num_low) / max(W - num_low, 1), 0.0, 1.0))
    rng = np.random.RandomState(seed)
    accel_mask = (rng.uniform(size=W) < prob).astype(np.float32)
    center_mask = np.zeros(W, dtype=np.float32)
    pad = (W - num_low + 1) // 2
    center_mask[pad:pad + num_low] = 1.0
    mask_1d = np.maximum(center_mask, accel_mask)
    return np.repeat(mask_1d[None, :], H, axis=0).astype(np.float32), num_low


def tv_recon_singlecoil(us_ks_centered, mask_2d, lam, num_iters=200, step_size=0.1, use_momentum=False):
    y = us_ks_centered.astype(np.complex64)
    mask = mask_2d.astype(np.float32)
    x = ifft2c_np(y)
    z, t = x.copy(), 1.0
    history, diverged, diverged_iter = [], False, None

    for i in range(num_iters):
        grad = ifft2c_np(mask * (fft2c_np(z) - y))
        z_step = z - step_size * grad
        if not np.all(np.isfinite(z_step)):
            diverged, diverged_iter = True, i
            print(f'Diverged before TV prox at iteration {i}')
            break
        x_r = denoise_tv_chambolle(z_step.real.astype(np.float32), weight=lam * step_size, channel_axis=None)
        x_i = denoise_tv_chambolle(z_step.imag.astype(np.float32), weight=lam * step_size, channel_axis=None)
        x_next = x_r.astype(np.complex64) + 1j * x_i.astype(np.complex64)
        if not np.all(np.isfinite(x_next)):
            diverged, diverged_iter = True, i
            print(f'Diverged after TV prox at iteration {i}')
            break
        if use_momentum:
            t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
            z = x_next + ((t - 1.0) / t_next) * (x_next - x)
            t = t_next
        else:
            z = x_next
        x = x_next
        if i in {0, 1, 2, 4, 9, 24, 49, 99, num_iters - 1}:
            history.append({'iter': i, 'max_abs': float(np.abs(x).max()),
                            'mean_abs': float(np.abs(x).mean())})

    return x, {'diverged': diverged, 'diverged_iter': diverged_iter, 'history': history,
                'step_size': step_size, 'use_momentum': use_momentum, 'lambda': lam}


def run_tv_lambda_sweep(full_ks, accel, lam_values, seed=1234, num_iters=200, step_size=0.1,
                        use_momentum=False, crop_shape=(320, 320), center_fractions=None):
    """
    Sweep TV regularisation weights on a single k-space slice (numpy complex array).
    Returns dict with gt_crop, zf_crop, zf_psnr, zf_ssim, and per-lambda results list.
    """
    ks_scale = max(float(np.abs(full_ks).max()), 1e-12)
    full_ks_norm = (full_ks / ks_scale).astype(np.complex64)
    gt_crop = _center_crop_np(np.abs(ifft2c_np(full_ks_norm)).astype(np.float32), crop_shape)

    print(f'R={accel} | ks_scale={ks_scale:.3e} | gt max={float(gt_crop.max()):.4f}')

    H, W = full_ks.shape
    cf = default_center_fraction(accel, center_fractions)
    mask_r, _ = _fastmri_mask_np(H, W, accel, cf, seed=seed)
    zf_crop = _center_crop_np(np.abs(ifft2c_np(full_ks_norm * mask_r)).astype(np.float32), crop_shape)

    results = []
    for lam in lam_values:
        recon_complex, info = tv_recon_singlecoil(
            full_ks_norm * mask_r, mask_r, lam=lam,
            num_iters=num_iters, step_size=step_size, use_momentum=use_momentum,
        )
        recon_crop = _center_crop_np(np.abs(recon_complex).astype(np.float32), crop_shape)
        results.append({'lambda': lam, 'recon_crop': recon_crop,
                        'psnr': psnr(gt_crop, recon_crop), 'ssim': ssim(gt_crop, recon_crop),
                        'info': info})
    return {'ks_scale': ks_scale, 'gt_crop': gt_crop, 'zf_crop': zf_crop,
            'zf_psnr': psnr(gt_crop, zf_crop), 'zf_ssim': ssim(gt_crop, zf_crop),
            'results': results}
