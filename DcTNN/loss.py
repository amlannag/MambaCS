"""Loss functions for MambaCS reconstruction."""
import math
import torch
import torch.nn.functional as F
from torch import nn
from normalizer import kspace_to_image_magnitude


_NORMALIZED_KSPACE_LOSS_NORMS = {"kspace_companding", "log_kspace"}


def _resolve_target(target, domain: str):
    if isinstance(target, dict):
        return target[domain]
    return target


def _use_normalized_kspace_loss(stats) -> bool:
    return bool(stats and stats.get("normalization") in _NORMALIZED_KSPACE_LOSS_NORMS)


def _to_magnitude(x, stats=None):
    """
    Bring x into the real magnitude image domain.
    - complex (k-space): ifft2 → abs → real magnitude, values ≥ 0
    - real (image domain): pass through as-is
    """
    if x.is_complex():
        if _use_normalized_kspace_loss(stats):
            return x.abs()
        return kspace_to_image_magnitude(x, stats)
    return x


def _gaussian_kernel(size: int = 11, sigma: float = 1.5, device=None) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel = g[:, None] * g[None, :]
    kernel /= kernel.sum()
    return kernel.view(1, 1, size, size)


def _radial_distance_grid_like(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 4:
        raise ValueError(f"Expected a [B, C, H, W] tensor, got shape {tuple(x.shape)}")
    _, _, h, w = x.shape
    ys = torch.arange(h, device=x.device, dtype=x.real.dtype if x.is_complex() else x.dtype)
    xs = torch.arange(w, device=x.device, dtype=ys.dtype)
    cy = h // 2
    cx = w // 2
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    dist = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    return dist.unsqueeze(0).unsqueeze(0)


def _perpendicular_mag_weight_map(x: torch.Tensor, m: float, k: float, p: float) -> torch.Tensor:
    dist = _radial_distance_grid_like(x)
    left = (m - 1.0) * torch.exp(k * (dist - p)) + 1.0
    ones = torch.ones_like(left)
    return torch.where(dist < p, left, ones)


class MagnitudeImageLoss(nn.Module):
    """MSE in the normalised magnitude image domain."""
    def forward(self, pred, gt, stats=None):
        target_domain = "kspace" if _use_normalized_kspace_loss(stats) else "image"
        gt_tensor = _resolve_target(gt, target_domain)
        return torch.mean((_to_magnitude(pred, stats) - _to_magnitude(gt_tensor, stats)) ** 2)


class MagnitudeL1Loss(nn.Module):
    """L1 loss in the normalised magnitude image domain."""
    def forward(self, pred, gt, stats=None):
        target_domain = "kspace" if _use_normalized_kspace_loss(stats) else "image"
        gt_tensor = _resolve_target(gt, target_domain)
        return torch.mean(torch.abs(_to_magnitude(pred, stats) - _to_magnitude(gt_tensor, stats)))


class ComplexL1Loss(nn.Module):
    """L1 loss directly in complex k-space: |Re diff| + |Im diff|."""
    target_domain = "kspace"

    def forward(self, pred, gt, stats=None):
        gt_kspace = _resolve_target(gt, self.target_domain)
        if not pred.is_complex() or not gt_kspace.is_complex():
            raise ValueError("complex_l1 requires complex k-space prediction and target tensors")
        return torch.mean(torch.abs(pred.real - gt_kspace.real) + torch.abs(pred.imag - gt_kspace.imag))


class ComplexL2Loss(nn.Module):
    """L2 loss directly in complex k-space: squared real diff + squared imag diff."""
    target_domain = "kspace"

    def forward(self, pred, gt, stats=None):
        gt_kspace = _resolve_target(gt, self.target_domain)
        if not pred.is_complex() or not gt_kspace.is_complex():
            raise ValueError("complex_l2 requires complex k-space prediction and target tensors")
        return torch.mean((pred.real - gt_kspace.real) ** 2 + (pred.imag - gt_kspace.imag) ** 2)


class PerpendicularLoss(nn.Module):
    """Perpendicular complex k-space loss with magnitude L1 term."""
    target_domain = "kspace"

    def __init__(
        self,
        eps: float = 1e-8,
        magnitude_weighting: bool = False,
        magnitude_weight_m: float = 1.0,
        magnitude_weight_k: float = 0.103,
        magnitude_weight_p: float = 67.0,
    ):
        super().__init__()
        self.eps = eps
        self.magnitude_weighting = magnitude_weighting
        self.magnitude_weight_k = float(magnitude_weight_k)
        self.magnitude_weight_p = float(magnitude_weight_p)
        self.current_m = float(magnitude_weight_m)

    def set_current_m(self, value: float):
        self.current_m = float(value)

    def forward(self, pred, gt, stats=None):
        gt_kspace = _resolve_target(gt, self.target_domain)
        if not pred.is_complex() or not gt_kspace.is_complex():
            raise ValueError("perpendicular_loss requires complex k-space prediction and target tensors")

        cross = pred * gt_kspace.conj()
        phi_hat = torch.angle(cross)
        perp = torch.abs(pred * gt_kspace.conj() - pred.conj() * gt_kspace) / (pred.abs() + self.eps)
        target_abs = gt_kspace.abs()
        branched = torch.where(phi_hat.abs() < (math.pi / 2), perp, 2 * target_abs - perp)
        magnitude_l1 = torch.abs(target_abs - pred.abs())
        if self.magnitude_weighting:
            magnitude_l1 = magnitude_l1 * _perpendicular_mag_weight_map(
                pred, m=self.current_m, k=self.magnitude_weight_k, p=self.magnitude_weight_p
            )
        return torch.mean(branched + magnitude_l1)


def build_loss(loss_type: str, **kwargs) -> nn.Module:
    """Factory for magnitude-domain reconstruction losses."""
    loss_type = loss_type.lower()
    if loss_type in {"l1", "image_domain_l1"}:
        return MagnitudeL1Loss()
    if loss_type in {"l2", "image_domain_l2"}:
        return MagnitudeImageLoss()
    if loss_type == "complex_l1":
        return ComplexL1Loss()
    if loss_type == "complex_l2":
        return ComplexL2Loss()
    if loss_type == "perpendicular_loss":
        return PerpendicularLoss(**kwargs)
    raise ValueError(
        "Unknown loss_type "
        f"'{loss_type}'. Choose from: ['l1', 'l2', 'image_domain_l1', 'image_domain_l2', "
        "'complex_l1', 'complex_l2', 'perpendicular_loss']"
    )


class SSIMLoss(nn.Module):
    """
    1 - SSIM, computed in the magnitude image domain.
    Uses an 11×11 Gaussian window (sigma=1.5), standard parameters.
    """

    def __init__(self, kernel_size: int = 11, sigma: float = 1.5):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def _ssim(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        kernel = _gaussian_kernel(self.kernel_size, self.sigma, device=x.device)
        pad = self.kernel_size // 2

        mu_x  = F.conv2d(x, kernel, padding=pad)
        mu_y  = F.conv2d(y, kernel, padding=pad)
        mu_xx = F.conv2d(x * x, kernel, padding=pad)
        mu_yy = F.conv2d(y * y, kernel, padding=pad)
        mu_xy = F.conv2d(x * y, kernel, padding=pad)

        sigma_x  = mu_xx - mu_x ** 2
        sigma_y  = mu_yy - mu_y ** 2
        sigma_xy = mu_xy - mu_x * mu_y

        num = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        den = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)
        return (num / den).mean()

    def forward(self, pred: torch.Tensor, gt: torch.Tensor, stats=None) -> torch.Tensor:
        gt_image = _resolve_target(gt, "image")
        p = _to_magnitude(pred, stats)
        g = _to_magnitude(gt_image)
        return 1.0 - self._ssim(p, g)
