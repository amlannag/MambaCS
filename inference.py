"""
Inference utilities for MambaCS experiments.
Provides load_experiment_model (used by notebooks) and denormalization helpers.
"""

import json
import os

import torch

from config import Config
from train_utils import build_model
from normalizer import kspace_to_image_magnitude, restore_original_kspace


# Fields that belong in cfg['data'] vs cfg['model'] for notebook consumers.
_DATA_KEYS = {
    "dataset", "data_dir", "val_data_dir", "kspace_key", "image_size",
    "num_channels", "acceleration_factors", "center_fractions", "mask_type",
    "val_fraction", "seed", "max_val_files", "norm", "companding_p", "companding_a",
}
_MODEL_KEYS = {
    "encoders", "patch_size", "axial_row_stride", "nhead_patch", "nhead_axial",
    "layer_no", "num_encoder_layers", "learned_lambda", "learning",
    "lambda_schedule", "lambda_start", "lambda_end",
    "pos_emb_type", "attn_type", "rope_theta", "rope_mixed_rotate",
    "mask_vertical_attn",
}


def _flat_to_nested(flat: dict) -> dict:
    """Split a flat config dict into {'data': {...}, 'model': {...}, 'train': {...}}."""
    data, model, train = {}, {}, {}
    for k, v in flat.items():
        if k in _DATA_KEYS:
            data[k] = v
        elif k in _MODEL_KEYS:
            model[k] = v
        else:
            train[k] = v
    return {"data": data, "model": model, "train": train}


def _flat_to_cfg(flat: dict) -> Config:
    """Reconstruct a Config dataclass from the flat dict saved by train.py."""
    cfg = Config()
    for k, v in flat.items():
        if not hasattr(cfg, k):
            continue
        if k in ("image_size", "patch_size") and isinstance(v, list):
            v = tuple(v)
        setattr(cfg, k, v)
    return cfg


def _migrate_complex_layernorm(state_dict: dict) -> dict:
    """
    Migrate checkpoints saved with the old ComplexLayerNorm (single complex gamma)
    to the new 2x2 covariance form (gamma_rr, gamma_ii, gamma_ri).

    Old form: gamma = complex tensor, applied as complex multiplication
        real_out = whitened.real * gamma.real - whitened.imag * gamma.imag
        imag_out = whitened.real * gamma.imag + whitened.imag * gamma.real
    New form: 2x2 matrix [[gamma_rr, gamma_ri], [gamma_ri, gamma_ii]]
    Mapping:  gamma_rr = gamma.real, gamma_ii = gamma.real, gamma_ri = -gamma.imag
              (approximation — whitening also changed from scalar to 2x2 covariance)
    """
    migrated = {}
    for k, v in state_dict.items():
        if k.endswith(".gamma") and torch.is_complex(v):
            base = k[: -len(".gamma")]
            migrated[base + ".gamma_rr"] = v.real.clone()
            migrated[base + ".gamma_ii"] = v.real.clone()
            migrated[base + ".gamma_ri"] = -v.imag.clone()
        else:
            migrated[k] = v
    return migrated


def load_experiment_model(exp_dir: str, device=None):
    """
    Load a trained experiment from an Experiments/<name>/ directory.

    Returns a dict with:
        'model':      the loaded nn.Module in eval mode
        'config':     nested dict {'data': {...}, 'model': {...}, 'train': {...}}
        'checkpoint': raw checkpoint metadata (epoch, val_psnr, best_val_psnr)
        'cfg':        Config dataclass (for code that prefers attribute access)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config_path = os.path.join(exp_dir, "config.json")
    ckpt_path = os.path.join(exp_dir, "best_model.pth")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json not found in {exp_dir}")

    with open(config_path) as f:
        flat = json.load(f)

    cfg = _flat_to_cfg(flat)
    model = build_model(cfg).to(device)

    ckpt_meta = {}
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        sd = _migrate_complex_layernorm(ckpt["model"])
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            print(f"  [WARN] {os.path.basename(exp_dir)}: "
                  f"{len(missing)} missing, {len(unexpected)} unexpected keys "
                  f"(ComplexLayerNorm migration applied)")
        ckpt_meta = {k: v for k, v in ckpt.items() if k != "model"}
    else:
        print(f"  WARNING — no checkpoint found at {ckpt_path}, using random weights")

    model.eval()

    return {
        "model":      model,
        "config":     _flat_to_nested(flat),
        "checkpoint": ckpt_meta,
        "cfg":        cfg,
    }


def _denormalize_image(recon: torch.Tensor, stats: dict) -> torch.Tensor:
    """
    Restore a reconstructed tensor toward original units.
    For normalized k-space modes this returns original-scale complex k-space.
    For z-score stats it preserves the previous real/complex affine restoration.
    """
    if recon.is_complex() and stats and stats.get("normalization") in {"kspace_companding", "log_kspace"}:
        return restore_original_kspace(recon, stats)
    if not stats or "mean_r" not in stats:
        return recon
    mean_r, std_r = stats["mean_r"], stats["std_r"]
    mean_i, std_i = stats["mean_i"], stats["std_i"]
    if recon.is_complex():
        return torch.complex(
            recon.real * std_r + mean_r,
            recon.imag * std_i + mean_i,
        )
    return recon * std_r + mean_r


def to_image_magnitude(recon: torch.Tensor, stats: dict | None = None) -> torch.Tensor:
    """Convert a reconstruction into image magnitude, restoring supported k-space norms first."""
    if recon.is_complex():
        return kspace_to_image_magnitude(recon, stats)
    return recon
