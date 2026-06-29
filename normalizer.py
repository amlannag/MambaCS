"""
Normalisation strategies for the MRI undersampling pipeline.

Each function takes (kspace_full, mask, learning) and returns
(model_input, DC_input, gt, metric) — identical contract to simulate_undersampling.

learning="k_space" : model_input is complex k-space  [B,1,H,W]
learning="image"   : model_input is real magnitude    [B,1,H,W]
"""

import torch
from DcTNN.dc import fft_2d, ifft_2d


def zscore(kspace_full, mask, learning="k_space", kspace_us=None):
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


def none(kspace_full, mask, learning="k_space", kspace_us=None):
    """No normalisation — tensors passed through in raw k-space units."""
    if kspace_us is None:
        kspace_us = kspace_full * mask
    img_us    = ifft_2d(kspace_us)
    img_gt    = ifft_2d(kspace_full)
    return _build_outputs(img_us, img_gt, {"normalization": "none"}, learning)


def _build_outputs(img_us_norm, img_gt_norm, metric, learning):
    gt = torch.abs(img_gt_norm)
    if learning == "k_space":
        DC_input    = fft_2d(img_us_norm)
        model_input = DC_input
    else:  # "image"
        model_input = torch.abs(img_us_norm)
        DC_input    = fft_2d(img_us_norm)   # actual measured k-space, not FFT of magnitude
    return model_input, DC_input, gt, metric
