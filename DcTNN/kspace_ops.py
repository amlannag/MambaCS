"""
Learnable k-space pre-processing stems applied to the encoder input before tokenisation.
Both are exact identities at initialisation so an existing recipe is unchanged until they learn.
"""
import math

import torch
import torch.nn as nn

from .complex_init import trabelsi_init_
from .util import ComplexGELU


class GlobalFilter(nn.Module):
    """
    Learnable complex Hadamard mask over the full k-space grid: x -> x * W, W in C^{H x W} initialised to 1.
    A per-cell multiplication in k-space is a circular convolution in the image domain, so this is a
    learned global image-domain convolution kernel (magnitude re-weighting plus per-cell phase rotation).
    """
    def __init__(self, image_size):
        super().__init__()
        h, w = (image_size, image_size) if isinstance(image_size, int) else tuple(image_size)
        self.weight = nn.Parameter(torch.ones(1, 1, h, w, dtype=torch.cfloat))

    def forward(self, x):
        return x * self.weight


class KSpaceConvStem(nn.Module):
    """
    Residual complex convolution over neighbouring k-space samples:
        x -> x + conv(c -> 1)(gelu(conv(1 -> c)(x)))
    Local mixing of adjacent k-space lines (a learned GRAPPA/LORAKS-style kernel), which in the image
    domain is a multiplicative modulation. The output conv is zero-initialised so the stem starts as identity.
    """
    def __init__(self, num_channels=1, hidden_channels=8, kernel_size=3):
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError(f"kspace_conv_kernel must be a positive odd integer, got {kernel_size}")
        pad = kernel_size // 2
        self.conv_in = nn.Conv2d(num_channels, hidden_channels, kernel_size, padding=pad, dtype=torch.cfloat)
        self.act = ComplexGELU()
        self.conv_out = nn.Conv2d(hidden_channels, num_channels, kernel_size, padding=pad, dtype=torch.cfloat)
        trabelsi_init_(self.conv_in.weight, fan_in=num_channels * kernel_size ** 2,
                       fan_out=hidden_channels * kernel_size ** 2, criterion="he")
        nn.init.zeros_(self.conv_in.bias)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, x):
        return x + self.conv_out(self.act(self.conv_in(x)))


def build_kspace_stem(image_size, num_channels=1, global_filter=False, kspace_conv=False,
                      kspace_conv_channels=8, kspace_conv_kernel=3):
    """
    Compose the enabled stems in the order GlobalFilter -> KSpaceConvStem. Returns None when neither is enabled
    so callers can skip the call entirely.
    """
    stages = []
    if global_filter:
        stages.append(GlobalFilter(image_size))
    if kspace_conv:
        stages.append(KSpaceConvStem(num_channels, kspace_conv_channels, kspace_conv_kernel))
    return nn.Sequential(*stages) if stages else None
