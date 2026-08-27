"""
Normalisation strategies for the MRI undersampling pipeline.

Each function takes (kspace_full, mask, learning) and returns
(model_input, DC_input, target, metric) — identical contract to simulate_undersampling.

learning="k_space"      : model_input is complex k-space [B,1,H,W]
learning="image"        : model_input is real magnitude   [B,1,H,W]
learning="complex_image": model_input is complex image    [B,1,H,W]
"""

import torch
from DcTNN.dc import fft_2d, ifft_2d

COMPANDING_EPS = 1e-6
LOG_KSPACE_EPS = 0.0


def _companding_axis(size: int, kspace: torch.Tensor, centering: str) -> torch.Tensor:
    if centering == "fft":
        return torch.arange(size, device=kspace.device, dtype=kspace.real.dtype) - size // 2
    if centering == "legacy":
        return torch.linspace(-size / 2, size / 2, size, device=kspace.device, dtype=kspace.real.dtype)
    raise ValueError("companding_centering must be 'fft' or 'legacy'")


def _companding_weight_like(
    kspace: torch.Tensor,
    a: float,
    p: float,
    eps: float = COMPANDING_EPS,
    centering: str = "fft",
) -> torch.Tensor:
    """Build a radial companding weight map centered on the shifted FFT origin."""
    _, _, h, w = kspace.shape
    y = _companding_axis(h, kspace, centering)
    x = _companding_axis(w, kspace, centering)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    weight = torch.clamp((a * (xx.square() + yy.square())).pow(p), min=eps)
    return weight.unsqueeze(0).unsqueeze(0)


def apply_kspace_companding(
    kspace: torch.Tensor,
    a: float,
    p: float,
    eps: float = COMPANDING_EPS,
    centering: str = "fft",
) -> torch.Tensor:
    """Scale k-space magnitude radially while preserving complex phase."""
    if not kspace.is_complex():
        raise TypeError("k-space companding requires a complex tensor")
    return kspace * _companding_weight_like(kspace, a=a, p=p, eps=eps, centering=centering)


def invert_kspace_companding(
    kspace: torch.Tensor,
    a: float,
    p: float,
    eps: float = COMPANDING_EPS,
    centering: str = "fft",
) -> torch.Tensor:
    """Invert radial k-space companding while preserving complex phase."""
    if not kspace.is_complex():
        raise TypeError("k-space companding inversion requires a complex tensor")
    return kspace / _companding_weight_like(kspace, a=a, p=p, eps=eps, centering=centering)


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


def smooth_clip(values: torch.Tensor, threshold: float) -> torch.Tensor:
    return values / torch.sqrt(1 + (values / threshold).square())


def invert_smooth_clip(values: torch.Tensor, threshold: float) -> torch.Tensor:
    limit = threshold * (1 - torch.finfo(values.dtype).eps)
    values = values.clamp(min=-limit, max=limit)
    return values / torch.sqrt(
        (1 - (values / threshold).square()).clamp_min(
            torch.finfo(values.dtype).eps
        )
    )


def _batch_stat_like(value, tensor: torch.Tensor) -> torch.Tensor:
    statistic = torch.as_tensor(value, device=tensor.device, dtype=tensor.real.dtype)
    if statistic.numel() == tensor.shape[0]:
        return statistic.reshape(tensor.shape[0], *([1] * (tensor.ndim - 1)))
    return statistic


def apply_normalization(tensor: torch.Tensor, metric: dict | None) -> torch.Tensor:
    if not metric:
        return tensor
    normalization = metric.get("normalization", "none")
    if normalization == "none":
        return tensor
    if normalization == "fastmri_magnitude":
        return tensor / _batch_stat_like(metric["p95"], tensor)
    if normalization == "kspace_companding":
        return apply_kspace_companding(
            tensor,
            a=metric["companding_a"],
            p=metric["companding_p"],
            eps=metric.get("companding_eps", COMPANDING_EPS),
            centering=metric.get("companding_centering", "legacy"),
        )
    if normalization == "log_kspace":
        return apply_log_kspace(tensor, eps=metric.get("log_eps", LOG_KSPACE_EPS))
    if normalization == "zscore":
        if tensor.is_complex():
            return torch.complex(
                (tensor.real - metric["mean_r"]) / metric["std_r"],
                (tensor.imag - metric["mean_i"]) / metric["std_i"],
            )
        return (tensor - metric["mean_r"]) / metric["std_r"]
    if normalization == "robust_shifted":
        coefficient = smooth_clip(
            metric["scale"] * (tensor.abs() - metric["median"]),
            metric["robust_clip"],
        )
        magnitude = coefficient + metric["robust_shift"]
        return torch.polar(magnitude, torch.angle(tensor)) if tensor.is_complex() else magnitude
    raise ValueError(f"Unknown normalization {normalization!r}")


def invert_normalization(tensor: torch.Tensor, metric: dict | None) -> torch.Tensor:
    if not metric:
        return tensor
    normalization = metric.get("normalization", "none")
    if normalization == "none":
        return tensor
    if normalization == "fastmri_magnitude":
        return tensor * _batch_stat_like(metric["p95"], tensor)
    if normalization == "kspace_companding":
        return invert_kspace_companding(
            tensor,
            a=metric["companding_a"],
            p=metric["companding_p"],
            eps=metric.get("companding_eps", COMPANDING_EPS),
            centering=metric.get("companding_centering", "legacy"),
        )
    if normalization == "log_kspace":
        return invert_log_kspace(tensor)
    if normalization == "zscore":
        if tensor.is_complex():
            return torch.complex(
                tensor.real * metric["std_r"] + metric["mean_r"],
                tensor.imag * metric["std_i"] + metric["mean_i"],
            )
        return tensor * metric["std_r"] + metric["mean_r"]
    if normalization == "robust_shifted":
        normalized_magnitude = tensor.abs() if tensor.is_complex() else tensor
        coefficient = normalized_magnitude - metric["robust_shift"]
        scaled = invert_smooth_clip(coefficient, metric["robust_clip"])
        scale = metric["scale"]
        safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        magnitude = torch.where(
            scale > 0,
            scaled / safe_scale + metric["median"],
            metric["median"],
        ).clamp_min(0)
        return torch.polar(magnitude, torch.angle(tensor)) if tensor.is_complex() else magnitude
    raise ValueError(f"Unknown normalization {normalization!r}")


def restore_original_kspace(kspace: torch.Tensor, metric: dict | None = None) -> torch.Tensor:
    """Map directly normalized complex k-space back to original k-space units."""
    if not kspace.is_complex() or not metric:
        return kspace
    normalization = metric.get("normalization", "none")
    direct_kspace_normalizations = {
        "kspace_companding",
        "log_kspace",
        "fastmri_magnitude",
        "robust_shifted",
    }
    if (
        metric.get("normalization_domain") == "k_space"
        or normalization in direct_kspace_normalizations
        and "normalization_domain" not in metric
    ):
        return invert_normalization(kspace, metric)
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


def complex_image_to_magnitude(image: torch.Tensor, metric: dict | None = None) -> torch.Tensor:
    if not image.is_complex():
        return image
    if metric and (
        metric.get("normalization_domain") == "complex_image"
        or metric.get("normalization") == "fastmri_magnitude"
        and "normalization_domain" not in metric
    ):
        image = invert_normalization(image, metric)
    return image.abs()


def _resolve_normalization_domain(metric: dict | None, learning: str) -> str:
    if metric and "normalization_domain" in metric:
        return metric["normalization_domain"]
    if metric and metric.get("normalization") in {"kspace_companding", "log_kspace"}:
        return "k_space"
    if metric and metric.get("normalization") == "zscore":
        return "complex_image"
    return learning


def model_output_to_raw_kspace(
    value: torch.Tensor, metric: dict | None, learning: str
) -> torch.Tensor:
    normalization_domain = _resolve_normalization_domain(metric, learning)
    if learning == "k_space":
        if normalization_domain == "k_space":
            return invert_normalization(value, metric)
        return fft_2d(invert_normalization(ifft_2d(value), metric))
    if learning in {"image", "complex_image"}:
        return fft_2d(invert_normalization(value, metric))
    raise ValueError(f"Unknown learning domain {learning!r}")


def raw_kspace_to_model_output(
    kspace: torch.Tensor, metric: dict | None, learning: str
) -> torch.Tensor:
    normalization_domain = _resolve_normalization_domain(metric, learning)
    if learning == "k_space":
        if normalization_domain == "k_space":
            return apply_normalization(kspace, metric)
        return fft_2d(apply_normalization(ifft_2d(kspace), metric))
    normalized_image = apply_normalization(ifft_2d(kspace), metric)
    if learning == "complex_image":
        return normalized_image
    if learning == "image":
        return normalized_image.abs()
    raise ValueError(f"Unknown learning domain {learning!r}")


def reconstruction_to_image_magnitude(recon: torch.Tensor, metric: dict | None = None) -> torch.Tensor:
    if not recon.is_complex():
        return recon
    learning = metric.get("prediction_domain", "k_space") if metric else "k_space"
    raw_kspace = model_output_to_raw_kspace(recon, metric, learning)
    return ifft_2d(raw_kspace).abs()


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
        "normalization_domain": "complex_image",
        "mean_r": mean_r,
        "std_r": std_r,
        "mean_i": mean_i,
        "std_i": std_i,
    }
    model_input, dc_input, target, metric = _build_outputs(
        img_us_norm, img_gt_norm, metric, learning, kspace_us
    )
    target["image"] = img_gt.abs()
    return model_input, dc_input, target, metric


def none(kspace_full, mask, learning="k_space", kspace_us=None, **_unused):
    """No normalisation — tensors passed through in raw k-space units."""
    if kspace_us is None:
        kspace_us = kspace_full * mask
    img_us    = ifft_2d(kspace_us)
    img_gt    = ifft_2d(kspace_full)
    metric = {"normalization": "none", "normalization_domain": learning}
    return _build_outputs(img_us, img_gt, metric, learning, kspace_us)


def kspace_companding(
    kspace_full,
    mask,
    learning="k_space",
    kspace_us=None,
    companding_p: float = 0.8,
    companding_a: float = 0.5,
    companding_eps: float = COMPANDING_EPS,
    companding_centering: str = "fft",
    **_unused,
):
    """Radially compand k-space magnitude while keeping the model entirely in companded k-space."""
    if learning != "k_space":
        raise ValueError("norm='kspace_companding' is only supported when learning='k_space'")
    if kspace_us is None:
        kspace_us = kspace_full * mask

    kspace_us_comp = apply_kspace_companding(
        kspace_us, a=companding_a, p=companding_p, eps=companding_eps, centering=companding_centering
    )
    kspace_full_comp = apply_kspace_companding(
        kspace_full, a=companding_a, p=companding_p, eps=companding_eps, centering=companding_centering
    )
    metric = {
        "normalization": "kspace_companding",
        "normalization_domain": "k_space",
        "prediction_domain": "k_space",
        "companding_p": companding_p,
        "companding_a": companding_a,
        "companding_eps": companding_eps,
        "companding_centering": companding_centering,
        "image_shape": tuple(int(v) for v in kspace_full.shape[-2:]),
    }
    target = {
        "image": kspace_to_image_magnitude(kspace_full_comp, metric),
        "kspace": kspace_full_comp,
    }
    return kspace_us_comp, kspace_us, target, metric


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
        "normalization_domain": "k_space",
        "prediction_domain": "k_space",
        "log_eps": log_eps,
        "image_shape": tuple(int(v) for v in kspace_full.shape[-2:]),
    }
    target = {
        "image": kspace_to_image_magnitude(kspace_full_log, metric),
        "kspace": kspace_full_log,
    }
    return kspace_us_log, kspace_us, target, metric


def _robust_location_scale(magnitudes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    quantile_levels = torch.tensor(
        [0.0, 0.25, 0.5, 0.75, 1.0],
        device=magnitudes.device,
        dtype=magnitudes.dtype,
    )
    q0, q25, median, q75, q100 = torch.quantile(
        magnitudes, quantile_levels, dim=1
    ).unbind(0)
    iqr = q75 - q25
    value_range = q100 - q0
    scale = torch.zeros_like(iqr)
    has_iqr = iqr > 0
    scale[has_iqr] = iqr[has_iqr].reciprocal()
    use_range = ~has_iqr & (value_range > 0)
    scale[use_range] = 2 * value_range[use_range].reciprocal()
    return median.reshape(-1, 1, 1, 1), scale.reshape(-1, 1, 1, 1)


def robust_shifted(
    kspace_full,
    mask,
    learning="k_space",
    kspace_us=None,
    robust_clip: float = 3.0,
    robust_shift: float = 3.0,
    **_unused,
):
    """Apply reversible median/IQR scaling, smooth clipping, and a magnitude shift."""
    if robust_clip <= 0:
        raise ValueError("robust_clip must be positive")
    if robust_shift < robust_clip:
        raise ValueError("robust_shift must be at least robust_clip")
    if kspace_us is None:
        kspace_us = kspace_full * mask

    if learning == "k_space":
        expanded_mask = mask.to(dtype=torch.bool).expand_as(kspace_us)
        measured_magnitudes = torch.stack(
            [
                kspace_us[index].abs()[expanded_mask[index]]
                for index in range(kspace_us.shape[0])
            ]
        )
        median, scale = _robust_location_scale(measured_magnitudes)
        metric = {
            "normalization": "robust_shifted",
            "normalization_domain": "k_space",
            "prediction_domain": "k_space",
            "median": median,
            "scale": scale,
            "robust_clip": float(robust_clip),
            "robust_shift": float(robust_shift),
        }
        kspace_us_norm = apply_normalization(kspace_us, metric) * expanded_mask
        kspace_full_norm = apply_normalization(kspace_full, metric)
        target = {
            "image": ifft_2d(kspace_full).abs(),
            "complex_image": ifft_2d(kspace_full_norm),
            "kspace": kspace_full_norm,
        }
        return kspace_us_norm, kspace_us, target, metric

    img_us = ifft_2d(kspace_us)
    img_gt = ifft_2d(kspace_full)
    median, scale = _robust_location_scale(
        img_us.abs().reshape(img_us.shape[0], -1)
    )
    metric = {
        "normalization": "robust_shifted",
        "normalization_domain": "complex_image",
        "median": median,
        "scale": scale,
        "robust_clip": float(robust_clip),
        "robust_shift": float(robust_shift),
    }
    img_us_norm = apply_normalization(img_us, metric)
    img_gt_norm = apply_normalization(img_gt, metric)
    model_input, dc_input, target, metric = _build_outputs(
        img_us_norm, img_gt_norm, metric, learning, kspace_us
    )
    target["image"] = img_gt.abs()
    return model_input, dc_input, target, metric


def fastmri_magnitude(kspace_full, mask, learning="k_space", kspace_us=None, **_unused):
    """Scale each sample by the undersampled magnitude p95 in its learning domain."""
    if kspace_us is None:
        kspace_us = kspace_full * mask

    if learning == "k_space":
        magnitudes = kspace_us.abs().reshape(kspace_us.shape[0], -1)
        scale_factor = torch.quantile(magnitudes, q=0.95, dim=1).clamp_min(1e-8)
        scale_factor = scale_factor.reshape(-1, 1, 1, 1)
        kspace_us_norm = kspace_us / scale_factor
        kspace_full_norm = kspace_full / scale_factor
        metric = {
            "normalization": "fastmri_magnitude",
            "normalization_domain": "k_space",
            "prediction_domain": "k_space",
            "p95": scale_factor,
        }
        target = {
            "image": ifft_2d(kspace_full).abs(),
            "complex_image": ifft_2d(kspace_full_norm),
            "kspace": kspace_full_norm,
        }
        return kspace_us_norm, kspace_us, target, metric

    img_us = ifft_2d(kspace_us)
    img_gt = ifft_2d(kspace_full)
    magnitudes = img_us.abs().reshape(img_us.shape[0], -1)
    scale_factor = torch.quantile(magnitudes, q=0.95, dim=1).clamp_min(1e-8)
    scale_factor = scale_factor.reshape(-1, 1, 1, 1)
    img_us_norm = img_us / scale_factor
    img_gt_norm = img_gt / scale_factor
    metric = {
        "normalization": "fastmri_magnitude",
        "normalization_domain": "complex_image",
        "p95": scale_factor,
    }
    model_input, dc_input, target, metric = _build_outputs(
        img_us_norm, img_gt_norm, metric, learning, kspace_us
    )
    target["image"] = img_gt.abs()
    return model_input, dc_input, target, metric


def _build_outputs(img_us_norm, img_gt_norm, metric, learning, kspace_us_raw):
    metric["prediction_domain"] = learning
    gt_kspace = fft_2d(img_gt_norm)
    target = {
        "image": torch.abs(img_gt_norm),
        "complex_image": img_gt_norm,
        "kspace": gt_kspace,
    }
    if learning == "k_space":
        model_input = fft_2d(img_us_norm)
    elif learning == "image":
        model_input = torch.abs(img_us_norm)
    elif learning == "complex_image":
        model_input = img_us_norm
    else:
        raise ValueError("learning must be 'k_space', 'image', or 'complex_image'")
    return model_input, kspace_us_raw, target, metric
