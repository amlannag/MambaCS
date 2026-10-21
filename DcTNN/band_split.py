"""
Band-split cascade stage: two experts combined by a radial Gaussian partition of unity.

    G_i(r) = exp(-(r / sigma_i)^2 / 2)             r = normalised k-space radius (0 at DC, 1 at the edge midpoints,
    out_i  = G_i * f_low(x) + (1 - G_i) * f_high(x)    sqrt2 in the corners; the grid used by the radial losses)

Both experts read the full current k-space; the partition only decides which expert's output is kept where, so the
low expert cannot write (and receives no gradient from) the high-frequency region and vice versa.

sigma is a cascade-level parameter vector (one value per freq_split stage) owned by `FreqSplitSigmas`, parameterised
so the boundaries are NON-DECREASING along the cascade: sigma_1 = lo + softplus(d_1), sigma_i = sigma_{i-1} + softplus(d_i),
clamped at the upper bound. Initialised from a linear / geometric schedule from sigma_start to sigma_end; learned or fixed.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

FREQ_SPLIT_SIGMA_SCHEDULES = ("linear", "geometric")
_SOFTPLUS_FLOOR = 1e-6   # smallest increment the parameterisation can represent (keeps the inverse softplus finite)


def normalized_radius_grid(image_size, device=None, dtype=torch.float32):
    """r on the normalised square: x = (col - W//2) / (W/2), y = (row - H//2) / (H/2). Same convention as DcTNN.loss."""
    h, w = (image_size, image_size) if isinstance(image_size, int) else tuple(image_size)
    ys = (torch.arange(h, device=device, dtype=dtype) - h // 2) / (h / 2)
    xs = (torch.arange(w, device=device, dtype=dtype) - w // 2) / (w / 2)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.sqrt(xx ** 2 + yy ** 2)


def gaussian_partition(radius, sigma):
    """Low-expert weight G = exp(-(r/sigma)^2 / 2); the high expert gets 1 - G."""
    return torch.exp(-0.5 * (radius / sigma) ** 2)


def freq_split_sigma_init(n_stages, start, end, schedule="geometric", bounds=(0.05, math.sqrt(2.0))):
    """Initial sigma per freq_split stage: `start` .. `end` spaced linearly or geometrically (a single stage gets `end`)."""
    if n_stages < 1:
        raise ValueError("freq_split needs at least one stage")
    if schedule not in FREQ_SPLIT_SIGMA_SCHEDULES:
        raise ValueError(f"freq_split_sigma_schedule must be one of {FREQ_SPLIT_SIGMA_SCHEDULES}, got {schedule!r}")
    if start <= 0 or end <= 0 or start > end:
        raise ValueError(f"need 0 < sigma_start <= sigma_end, got {start}, {end}")
    if n_stages == 1:
        values = np.array([end], dtype=np.float64)
    elif schedule == "linear":
        values = np.linspace(start, end, n_stages)
    else:
        values = np.geomspace(start, end, n_stages)
    lo, hi = bounds
    return [float(v) for v in np.clip(values, lo, hi)]


def _inverse_softplus(y):
    y = torch.as_tensor(y, dtype=torch.float64).clamp_min(_SOFTPLUS_FLOOR)
    return (y + torch.log(-torch.expm1(-y))).to(torch.float32)


class FreqSplitSigmas(nn.Module):
    """
    Non-decreasing sigma vector for the freq_split stages of a cascade.

    sigma_1 = lo + softplus(d_1);  sigma_i = sigma_{i-1} + softplus(d_i);  all clamped to `hi`.
    `sigma_init` must itself be non-decreasing and within (lo, hi]. With learned=False the raw increments are a buffer.
    """
    def __init__(self, sigma_init, learned=True, bounds=(0.05, math.sqrt(2.0))):
        super().__init__()
        lo, hi = float(bounds[0]), float(bounds[1])
        if not 0 < lo < hi:
            raise ValueError(f"need 0 < lower bound < upper bound, got {bounds}")
        sigma_init = [float(s) for s in sigma_init]
        if not sigma_init:
            raise ValueError("sigma_init must not be empty")
        if any(b < a for a, b in zip(sigma_init, sigma_init[1:])):
            raise ValueError(f"freq_split sigma initialisation must be non-decreasing along the cascade, got {sigma_init}")
        if sigma_init[0] <= lo or sigma_init[-1] > hi + 1e-6:
            raise ValueError(f"sigma_init must lie in ({lo}, {hi}], got {sigma_init}")
        increments = torch.tensor([sigma_init[0] - lo] + [b - a for a, b in zip(sigma_init, sigma_init[1:])])
        raw = _inverse_softplus(increments)
        self.lo, self.hi = lo, hi
        self.learned = bool(learned)
        if self.learned:
            self.raw = nn.Parameter(raw)
        else:
            self.register_buffer("raw", raw)

    def forward(self):
        return (self.lo + torch.cumsum(F.softplus(self.raw), dim=0)).clamp(max=self.hi)

    def values(self):
        return [float(v) for v in self.forward().detach().cpu()]

    def __len__(self):
        return self.raw.shape[0]

    def extra_repr(self):
        return f"n={len(self)}, learned={self.learned}, bounds=({self.lo}, {self.hi:.4f}), init={[round(v, 4) for v in self.values()]}"


class BandSplitVIT(nn.Module):
    """
    One freq_split cascade stage: `low` and `high` are complete stage modules (axVIT, NeighbourhoodVIT, ...) with the
    usual forward(x, col_mask=None) -> residual signature; this module returns G * low(x) + (1 - G) * high(x).
    sigma is supplied per call by the cascade (see FreqSplitSigmas) so the boundaries stay ordered across stages.
    """
    def __init__(self, N, low, high):
        super().__init__()
        self.N = N
        self.low = low
        self.high = high
        self.register_buffer("radius", normalized_radius_grid(N), persistent=False)

    def partition(self, sigma):
        """[1, 1, H, W] low-expert weight G for this sigma (float scalar or 0-d tensor)."""
        return gaussian_partition(self.radius, sigma).unsqueeze(0).unsqueeze(0)

    def forward(self, x, col_mask=None, sigma=None):
        if sigma is None:
            raise ValueError("BandSplitVIT.forward needs the stage sigma (provided by cascadeNet)")
        g = self.partition(sigma).to(x.real.dtype if x.is_complex() else x.dtype)
        return g * self.low(x, col_mask=col_mask) + (1.0 - g) * self.high(x, col_mask=col_mask)
