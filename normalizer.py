"""
Normalisation strategies for the MRI undersampling pipeline.

Each function takes (kspace_full, mask, learning) and returns
(model_input, DC_input, target, metric) — identical contract to simulate_undersampling.

learning="k_space" : model_input is complex k-space  [B,1,H,W]
learning="image"   : model_input is real magnitude    [B,1,H,W]
"""

import torch
from DcTNN.dc import fft_2d, ifft_2d

COMPANDING_EPS = 1e-6
LOG_KSPACE_EPS = 0.0


def _companding_weight_like(kspace: torch.Tensor, a: float, p: float, eps: float = COMPANDING_EPS) -> torch.Tensor:
    """Build a centered radial companding weight map matching the notebook convention."""
    _, _, h, w = kspace.shape
    y = torch.linspace(-h / 2, h / 2, h, device=kspace.device, dtype=kspace.real.dtype)
    x = torch.linspace(-w / 2, w / 2, w, device=kspace.device, dtype=kspace.real.dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    weight = torch.clamp((a * (xx.square() + yy.square())).pow(p), min=eps)
    return weight.unsqueeze(0).unsqueeze(0)


def apply_kspace_companding(kspace: torch.Tensor, a: float, p: float, eps: float = COMPANDING_EPS) -> torch.Tensor:
    """Scale k-space magnitude radially while preserving complex phase."""
    if not kspace.is_complex():
        raise TypeError("k-space companding requires a complex tensor")
    return kspace * _companding_weight_like(kspace, a=a, p=p, eps=eps)


def invert_kspace_companding(kspace: torch.Tensor, a: float, p: float, eps: float = COMPANDING_EPS) -> torch.Tensor:
    """Invert radial k-space companding while preserving complex phase."""
    if not kspace.is_complex():
        raise TypeError("k-space companding inversion requires a complex tensor")
    return kspace / _companding_weight_like(kspace, a=a, p=p, eps=eps)


def apply_log_kspace(kspace: torch.Tensor, eps: float = LOG_KSPACE_EPS) -> torch.Tensor:
    """Compress k-space magnitudes with log1p while preserving phase."""
    if not kspace.is_complex():
        raise TypeError("log-k-space normalization requires a complex tensor")
    magnitude = torch.log1p(kspace.abs().clamp_min(eps))
    phase = torch.angle(kspace)
    return torch.polar(magnitude, phase)


def invert_log_kspace(kspace: torch.Tensor) -> torch.Tensor:
    """Invert log-k-space normalization while preserving phase."""
    if not kspace.is_complex():
        raise TypeError("log-k-space inversion requires a complex tensor")
    magnitude = torch.expm1(kspace.abs())
    phase = torch.angle(kspace)
    return torch.polar(magnitude, phase)


def restore_original_kspace(kspace: torch.Tensor, metric: dict | None = None) -> torch.Tensor:
    """Map normalized complex k-space back to original k-space units when supported."""
    if not kspace.is_complex() or not metric:
        return kspace

    normalization = metric.get("normalization")
    if normalization == "kspace_companding":
        return invert_kspace_companding(
            kspace,
            a=metric["companding_a"],
            p=metric["companding_p"],
            eps=metric.get("companding_eps", COMPANDING_EPS),
        )
    if normalization == "log_kspace":
        return invert_log_kspace(kspace)
    return kspace


def kspace_to_image_magnitude(kspace: torch.Tensor, metric: dict | None = None) -> torch.Tensor:
    """
    Convert complex k-space into image magnitude.
    Normalized tensors are first restored to original k-space units when supported,
    then transformed via IFFT.
    """
    if not kspace.is_complex():
        return kspace
    kspace = restore_original_kspace(kspace, metric)
    return torch.abs(ifft_2d(kspace))


def zscore(kspace_full, mask, learning="k_space", kspace_us=None, **_unused):
    """Z-score normalise real and imaginary channels separately using undersampled image stats."""
    if kspace_us is None:
        kspace_us = kspace_full * mask
    img_us    = ifft_2d(kspace_us)
    img_gt    = ifft_2d(kspace_full)

    mean_r = img_us.real.mean(dim=(-2, -1), keepdim=True)
    std_r  = img_us.real.std( dim=(-2, -1), keepdim=True).clamp(min=1e-8)
    mean_i = img_us.imag.mean(dim=(-2, -1), keepdim=True)
    std_i  = img_us.imag.std( dim=(-2, -1), keepdim=True).clamp(min=1e-8)

    img_us_norm = torch.complex((img_us.real - mean_r) / std_r,
                                (img_us.imag - mean_i) / std_i)
    img_gt_norm = torch.complex((img_gt.real - mean_r) / std_r,
                                (img_gt.imag - mean_i) / std_i)

    metric = {
        "normalization": "zscore",
        "mean_r": mean_r,
        "std_r": std_r,
        "mean_i": mean_i,
        "std_i": std_i,
    }
    return _build_outputs(img_us_norm, img_gt_norm, metric, learning)


def none(kspace_full, mask, learning="k_space", kspace_us=None, **_unused):
    """No normalisation — tensors passed through in raw k-space units."""
    if kspace_us is None:
        kspace_us = kspace_full * mask
    img_us    = ifft_2d(kspace_us)
    img_gt    = ifft_2d(kspace_full)
    return _build_outputs(img_us, img_gt, {"normalization": "none"}, learning)


def kspace_companding(
    kspace_full,
    mask,
    learning="k_space",
    kspace_us=None,
    companding_p: float = 0.8,
    companding_a: float = 0.5,
    companding_eps: float = COMPANDING_EPS,
):
    """Radially compand k-space magnitude while keeping the model entirely in companded k-space."""
    if learning != "k_space":
        raise ValueError("norm='kspace_companding' is only supported when learning='k_space'")
    if kspace_us is None:
        kspace_us = kspace_full * mask

    kspace_us_comp = apply_kspace_companding(kspace_us, a=companding_a, p=companding_p, eps=companding_eps)
    kspace_full_comp = apply_kspace_companding(kspace_full, a=companding_a, p=companding_p, eps=companding_eps)
    metric = {
        "normalization": "kspace_companding",
        "companding_p": companding_p,
        "companding_a": companding_a,
        "companding_eps": companding_eps,
        "image_shape": tuple(int(v) for v in kspace_full.shape[-2:]),
    }
    target = {
        "image": kspace_to_image_magnitude(kspace_full_comp, metric),
        "kspace": kspace_full_comp,
    }
    return kspace_us_comp, kspace_us_comp, target, metric


def log_kspace(
    kspace_full,
    mask,
    learning="k_space",
    kspace_us=None,
    log_eps: float = LOG_KSPACE_EPS,
    **_unused,
):
    """Apply log1p to k-space magnitudes while preserving phase."""
    if learning != "k_space":
        raise ValueError("norm='log_kspace' is only supported when learning='k_space'")
    if kspace_us is None:
        kspace_us = kspace_full * mask

    kspace_us_log = apply_log_kspace(kspace_us, eps=log_eps)
    kspace_full_log = apply_log_kspace(kspace_full, eps=log_eps)
    metric = {
        "normalization": "log_kspace",
        "log_eps": log_eps,
        "image_shape": tuple(int(v) for v in kspace_full.shape[-2:]),
    }
    target = {
        "image": kspace_to_image_magnitude(kspace_full_log, metric),
        "kspace": kspace_full_log,
    }
    return kspace_us_log, kspace_us_log, target, metric


def fastmri_magnitude(kspace_full, mask, learning="k_space", kspace_us=None, **_unused):
    """
    FastMRI magnitude normalization: scale by 95th percentile of undersampled image magnitude.
    Computes the 95th percentile from the undersampled (masked) k-space in image domain,
    then applies it to both model input and GT for consistent loss calculation.
    Per-scan normalization ensures each image is independently scaled.
    """
    if kspace_us is None:
        kspace_us = kspace_full * mask

    # Convert to image domain
    img_us = ifft_2d(kspace_us)
    img_gt = ifft_2d(kspace_full)

    # Get magnitude of undersampled image in image domain
    mag_us = torch.abs(img_us)

    # Calculate 95th percentile per scan (batch-wise)
    # Flatten spatial dimensions while keeping batch dimension: [B, H*W]
    mag_flat = mag_us.reshape(mag_us.shape[0], -1)
    p95 = torch.quantile(mag_flat, q=0.95, dim=1)  # [B]

    # Reshape for broadcasting: [B, 1, 1, 1]
    p95 = p95.reshape(-1, 1, 1, 1)

    # Avoid division by zero
    scale_factor = p95.clamp(min=1e-8)

    # Scale both undersampled and GT by the same 95th percentile
    img_us_norm = img_us / scale_factor
    img_gt_norm = img_gt / scale_factor

    metric = {
        "normalization": "fastmri_magnitude",
        "p95": p95.squeeze(),
    }
    return _build_outputs(img_us_norm, img_gt_norm, metric, learning)


def _build_outputs(img_us_norm, img_gt_norm, metric, learning):
    gt_image = torch.abs(img_gt_norm)
    gt_kspace = fft_2d(img_gt_norm)
    target = {
        "image": gt_image,
        "kspace": gt_kspace,
    }
    if learning == "k_space":
        DC_input    = fft_2d(img_us_norm)
        model_input = DC_input
    else:  # "image"
        model_input = torch.abs(img_us_norm)
        DC_input    = fft_2d(img_us_norm)   # actual measured k-space, not FFT of magnitude
    return model_input, DC_input, target, metric
