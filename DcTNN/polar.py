"""
Cartesian <-> polar resampling of k-space for radial (spoke) tokens.

A spoke is a half-line from DC at angle theta_i = 2*pi*i / num_spokes; spokes i and i + num_spokes/2 are the two halves
of one line through DC. Sample j of a spoke sits at radius r_j = j * spacing, j = 0 .. num_radii-1, with
r_max = min(H, W) / 2 - 1 (inscribed circle: every sample lies on the grid, the corners outside the circle are not
covered).
Each polar sample is the bilinear interpolation of its 4 Cartesian neighbours, so Cartesian -> polar is a fixed sparse
linear map P (4 weights per polar sample) and polar -> Cartesian is the density-normalised adjoint D * P^T
(D = 1 / (P^T 1), i.e. a weighted average of the polar samples whose footprint covers the pixel; pixels no spoke
touches get 0). Both directions are implemented with gather / index_add_ on the real view, so they work for complex
tensors on any backend without sparse kernels.

Note (see notebooks/spoke_tokenisation.ipynb): D * P^T * P is NOT close to the identity for k-space of an object that
fills the FOV (bilinear interpolation apodises the image), so a radial pass must be used as a residual on the Cartesian
state, never as a replacement for it.
"""
import math

import torch
from torch import nn


def pair(t):
    return tuple(t) if isinstance(t, (tuple, list)) else (t, t)


class PolarResampler(nn.Module):
    def __init__(self, image_size, num_spokes=320, spacing=1.0):
        super().__init__()
        H, W = pair(image_size)
        if num_spokes < 2 or num_spokes % 2:
            raise ValueError(f"num_spokes must be an even number >= 2 (half-spokes pair up into lines), got {num_spokes}")
        if spacing <= 0:
            raise ValueError(f"spacing must be positive, got {spacing}")
        self.image_size = (H, W)
        self.num_spokes = int(num_spokes)
        self.spacing = float(spacing)
        r_max = min(H, W) / 2 - 1                      # DC sits at index H // 2, so the grid extends H // 2 - 1 past it
        self.num_radii = int(math.floor(r_max / spacing)) + 1
        cy, cx = H // 2, W // 2
        theta = torch.arange(self.num_spokes, dtype=torch.float64) * (2 * math.pi / self.num_spokes)
        radius = torch.arange(self.num_radii, dtype=torch.float64) * spacing
        yy = cy + radius[None, :] * torch.sin(theta)[:, None]                        # (spokes, radii)
        xx = cx + radius[None, :] * torch.cos(theta)[:, None]
        y0, x0 = yy.floor(), xx.floor()
        wy, wx = yy - y0, xx - x0
        polar_index = torch.arange(self.num_spokes * self.num_radii).view(self.num_spokes, self.num_radii)
        rows, cols, vals = [], [], []
        for dy, dx, weight in ((0, 0, (1 - wy) * (1 - wx)), (0, 1, (1 - wy) * wx),
                               (1, 0, wy * (1 - wx)), (1, 1, wy * wx)):
            yi, xi = (y0 + dy).long(), (x0 + dx).long()
            keep = (yi >= 0) & (yi < H) & (xi >= 0) & (xi < W) & (weight > 0)
            rows.append(polar_index[keep])
            cols.append((yi * W + xi)[keep])
            vals.append(weight[keep])
        rows, cols, vals = torch.cat(rows), torch.cat(cols), torch.cat(vals).float()
        density = torch.zeros(H * W).index_add_(0, cols, vals)
        self.register_buffer("rows", rows, persistent=False)
        self.register_buffer("cols", cols, persistent=False)
        self.register_buffer("vals", vals, persistent=False)
        self.register_buffer("inv_density", torch.where(density > 0, 1.0 / density.clamp_min(1e-12), torch.zeros_like(density)),
                             persistent=False)
        self.register_buffer("theta", theta.float(), persistent=False)
        self.register_buffer("radius", radius.float(), persistent=False)

    @property
    def covered_fraction(self):
        """Fraction of Cartesian pixels touched by at least one polar sample."""
        return float((self.inv_density > 0).float().mean())

    def _apply(self, x, src, dst, n_out, scale=None):
        """y[..., dst] += vals * x[..., src]  over the last axis; complex-safe via the real view."""
        is_complex = x.is_complex()
        xr = torch.view_as_real(x) if is_complex else x.unsqueeze(-1)              # (..., n_in, 2|1)
        lead = xr.shape[:-2]
        gathered = xr[..., src, :] * self.vals.view(*([1] * len(lead)), -1, 1)
        out = xr.new_zeros(*lead, n_out, xr.shape[-1])
        out.index_add_(len(lead), dst, gathered)
        if scale is not None:
            out = out * scale.view(*([1] * len(lead)), -1, 1)
        return torch.view_as_complex(out.contiguous()) if is_complex else out.squeeze(-1)

    def forward(self, kspace):
        """(B, C, H, W) -> (B, C, num_spokes, num_radii) polar samples."""
        B, C, H, W = kspace.shape
        if (H, W) != self.image_size:
            raise ValueError(f"expected {self.image_size} k-space, got {(H, W)}")
        polar = self._apply(kspace.reshape(B, C, H * W), self.cols, self.rows, self.num_spokes * self.num_radii)
        return polar.view(B, C, self.num_spokes, self.num_radii)

    def adjoint(self, polar):
        """(B, C, num_spokes, num_radii) -> (B, C, H, W): density-normalised adjoint (weighted average per pixel)."""
        B, C = polar.shape[:2]
        H, W = self.image_size
        cart = self._apply(polar.reshape(B, C, -1), self.rows, self.cols, H * W, scale=self.inv_density)
        return cart.view(B, C, H, W)

    def spoke_coordinates(self, radius=None):
        """(x, y) of a point on every spoke at `radius` (default: arc spacing of 1 between neighbouring spokes), for RoPE.
        Placing spokes on a circle makes the positional encoding periodic: spoke num_spokes-1 neighbours spoke 0."""
        if radius is None:
            radius = self.num_spokes / (2 * math.pi)
        return radius * torch.cos(self.theta), radius * torch.sin(self.theta)
