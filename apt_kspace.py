import math

import torch
import torch.nn.functional as F


def entropy_maps(score_image, base_patch_size=16, num_scales=3, bins=512):
    maps = {}
    for level in range(num_scales):
        size = base_patch_size * 2 ** level
        patches = score_image.unfold(0, size, size).unfold(1, size, size)
        flat = patches.reshape(-1, size * size)
        histograms = torch.stack([torch.histc(patch, bins=bins, min=0, max=255) for patch in flat])
        probabilities = histograms / (size * size)
        entropy = -(probabilities * torch.log2(probabilities + 1e-10)).sum(dim=1)
        maps[size] = entropy.reshape(patches.shape[:2])
    return maps


def select_patches(maps, thresholds):
    sizes = sorted(maps)
    if len(thresholds) != len(sizes) - 1:
        raise ValueError("One threshold is required for each scale above the base patch size")
    masks = {sizes[0]: torch.ones_like(maps[sizes[0]], dtype=torch.bool)}
    for size, threshold in zip(sizes[1:], thresholds):
        masks[size] = maps[size] < threshold
    for level in range(len(sizes) - 1, 0, -1):
        size = sizes[level]
        for smaller in sizes[:level]:
            factor = size // smaller
            covered = masks[size].repeat_interleave(factor, 0).repeat_interleave(factor, 1)
            masks[smaller] &= ~covered
    return masks


@torch.no_grad()
def tokenize_kspace(kspace, base_patch_size=16, num_scales=3, thresholds=(3.0, 3.0),
                    bins=512, representation="log", log_gain=1000.0, normalization="max"):
    if kspace.ndim != 2 or not kspace.is_complex() or not torch.isfinite(kspace).all():
        raise ValueError("Expected one finite complex [H, W] k-space slice")
    if not isinstance(base_patch_size, int) or base_patch_size < 1:
        raise ValueError("base_patch_size must be a positive integer")
    if not isinstance(num_scales, int) or num_scales < 1:
        raise ValueError("num_scales must be a positive integer")
    largest = base_patch_size * 2 ** (num_scales - 1)
    if any(dimension % largest for dimension in kspace.shape):
        raise ValueError(f"K-space dimensions must be divisible by the largest patch size ({largest})")
    if not isinstance(bins, int) or bins < 2:
        raise ValueError("bins must be an integer >= 2")
    if len(thresholds) != num_scales - 1 or not all(math.isfinite(t) for t in thresholds):
        raise ValueError("Supply num_scales - 1 finite thresholds, ordered from small to large scales")
    if representation not in {"linear", "log"}:
        raise ValueError("representation must be 'linear' or 'log'")
    if not math.isfinite(log_gain) or log_gain <= 0:
        raise ValueError("log_gain must be positive and finite")
    if normalization == "fastmri_magnitude":
        scale = torch.quantile(kspace.abs().reshape(-1), q=0.95).clamp_min(1e-8)
        relative = (kspace / scale).abs().float()
    elif normalization == "max":
        magnitude = kspace.abs().float()
        scale = magnitude.max()
        relative = magnitude / scale if scale > 0 else torch.zeros_like(magnitude)
    else:
        raise ValueError("normalization must be 'max' or 'fastmri_magnitude'")
    score = relative if representation == "linear" else torch.log1p(log_gain * relative) / math.log1p(log_gain)
    score = (255 * score).clamp(0, 255)
    maps = entropy_maps(score, base_patch_size, num_scales, bins)
    masks = select_patches(maps, thresholds)
    boxes = [(int(y) * size, int(x) * size, size)
             for size in sorted(masks) for y, x in masks[size].nonzero().tolist()]
    counts = {size: int(mask.sum()) for size, mask in masks.items()}
    baseline = kspace.numel() // base_patch_size ** 2
    return {
        "score_image": score,
        "importance_maps": maps,
        "masks": masks,
        "boxes": boxes,
        "counts": counts,
        "token_count": len(boxes),
        "baseline_tokens": baseline,
        "reduction_percent": 100 * (1 - len(boxes) / baseline),
        "base_patch_size": base_patch_size,
        "shape": tuple(kspace.shape),
        "representation": representation,
        "normalization": normalization,
        "normalization_scale": float(scale),
        "score_clipped_fraction": float((relative > 1).float().mean()),
        "thresholds": tuple(thresholds),
    }


def extract_patch_groups(kspace, result):
    if tuple(kspace.shape) != result["shape"] or not kspace.is_complex():
        raise ValueError("Complex k-space shape must match the tokenization result")
    base = result["base_patch_size"]
    groups = {}
    for size in sorted(result["masks"]):
        origins = [(y, x) for y, x, patch_size in result["boxes"] if patch_size == size]
        patches = (torch.stack([kspace[y:y + size, x:x + size] for y, x in origins])
                   if origins else kspace.new_empty((0, size, size)))
        resized = patches
        if size != base:
            if origins:
                channels = torch.stack((patches.real, patches.imag), dim=1)
                small = F.interpolate(channels, size=(base, base), mode="bilinear", align_corners=False)
                resized = torch.complex(small[:, 0], small[:, 1])
            else:
                resized = kspace.new_empty((0, base, base))
        constituents = patches.reshape(-1, size // base, base, size // base, base)
        constituents = constituents.permute(0, 1, 3, 2, 4).reshape(len(origins), (size // base) ** 2, base, base)
        groups[size] = {"origins": origins, "original": patches, "resized": resized,
                        "constituents": constituents}
    return groups
