"""
Data-consistency layers with spatially varying learned weights.

All three share the same blend as KSpace_DC:

    x_dc = (1 - m) * x + m * (x + lamb * y) / (1 + lamb)

where `lamb` is a scalar (ScalarLambdaDC, the current behaviour), one weight per sampled k-space pixel
(PixelwiseLambdaDC), or a coarse grid of weights spread evenly over the sampled pixels and bilinearly interpolated
to every sampled pixel (InterpolatedLambdaDC, 1/4 the parameters of PixelwiseLambdaDC by default).

Weights are stored as unconstrained raw parameters and mapped through softplus so lamb > 0; the raw values are
initialised such that every lamb starts at `init` (0.5, the same start as the scalar model lambda).
"""
import math

import torch
from torch import nn
import torch.nn.functional as F

from .dc import _to_complex_channel


def _softplus_inverse(value):
    return math.log(math.expm1(value))


def dc_blend(x, y, mask, lamb):
    """KSpace_DC with a scalar or broadcastable [H, W] / [1, 1, H, W] lamb."""
    x = _to_complex_channel(x)
    y = _to_complex_channel(y)
    return (1 - mask) * x + mask * (x + lamb * y) / (1 + lamb)


def sampled_columns(mask):
    """Column indices sampled by a Cartesian line mask (mask constant along the readout axis)."""
    mask2d = mask.reshape(-1, mask.shape[-1]) if mask.ndim > 2 else mask
    columns = (mask2d != 0).any(dim=0) if mask2d.shape[0] > 1 else (mask2d[0] != 0)
    if mask2d.shape[0] > 1 and not torch.equal((mask2d != 0), columns.expand_as(mask2d)):
        raise ValueError("Mask must be a Cartesian line mask (identical for every row)")
    return columns.nonzero().flatten()


class ScalarLambdaDC(nn.Module):
    """The current data consistency: one learned lambda (init 0.5)."""

    def __init__(self, init=0.5):
        super().__init__()
        self.lamb = nn.Parameter(torch.tensor(float(init)))

    def lambda_map(self, height, width):
        return self.lamb.expand(height, width)

    def forward(self, x, y, mask):
        return dc_blend(x, y, mask, self.lamb)

    @property
    def num_weights(self):
        return 1


class PixelwiseLambdaDC(nn.Module):
    """One learned lambda for every sampled k-space pixel (H x n_sampled_columns parameters)."""

    def __init__(self, mask, height, init=0.5):
        super().__init__()
        columns = sampled_columns(mask)
        self.register_buffer("columns", columns)
        self.height, self.width = int(height), int(mask.shape[-1])
        self.raw = nn.Parameter(torch.full((self.height, columns.numel()), _softplus_inverse(init)))

    def lambda_map(self, height=None, width=None):
        """Dense [H, W] lambda map (zeros at unsampled pixels, where lamb has no effect)."""
        full = self.raw.new_zeros(self.height, self.width)
        full[:, self.columns] = F.softplus(self.raw)
        return full

    def forward(self, x, y, mask):
        return dc_blend(x, y, mask, self.lambda_map())

    @property
    def num_weights(self):
        return self.raw.numel()


class InterpolatedLambdaDC(nn.Module):
    """
    A coarse grid of learned lambdas placed evenly over the sampled region — `reduction` x fewer weights along each
    of the (readout, sampled-column) axes, i.e. 1/reduction**2 of PixelwiseLambdaDC with the default 2 — bilinearly
    interpolated (align_corners=True, so the control points sit on the first/last row and sampled column) to every
    sampled pixel.
    """

    def __init__(self, mask, height, init=0.5, reduction=2):
        super().__init__()
        columns = sampled_columns(mask)
        self.register_buffer("columns", columns)
        self.height, self.width = int(height), int(mask.shape[-1])
        self.grid_shape = (max(2, math.ceil(self.height / reduction)), max(2, math.ceil(columns.numel() / reduction)))
        self.raw = nn.Parameter(torch.full(self.grid_shape, _softplus_inverse(init)))

    def sampled_lambda(self):
        """[H, n_sampled] lambdas: coarse grid -> bilinear interpolation over the sampled-pixel index space."""
        coarse = F.softplus(self.raw)[None, None]
        return F.interpolate(coarse, size=(self.height, self.columns.numel()), mode="bilinear", align_corners=True)[0, 0]

    def lambda_map(self, height=None, width=None):
        full = self.raw.new_zeros(self.height, self.width)
        full[:, self.columns] = self.sampled_lambda()
        return full

    def forward(self, x, y, mask):
        return dc_blend(x, y, mask, self.lambda_map())

    @property
    def num_weights(self):
        return self.raw.numel()


def build_dc(kind, mask, height, init=0.5, reduction=2):
    if kind == "scalar":
        return ScalarLambdaDC(init)
    if kind == "pixelwise":
        return PixelwiseLambdaDC(mask, height, init)
    if kind == "interpolated":
        return InterpolatedLambdaDC(mask, height, init, reduction)
    raise ValueError(f"Unknown DC kind {kind!r}; choose scalar, pixelwise or interpolated")
