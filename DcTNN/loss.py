"""Loss functions for MambaCS reconstruction."""
import torch
import torch.nn.functional as F
from torch import nn
from DcTNN.dc import ifft_2d


def _to_magnitude(x):
    """
    Bring x into the real magnitude image domain.
    - complex (k-space): ifft2 → abs → real magnitude, values ≥ 0
    - real (image domain): pass through as-is
    """
    if x.is_complex():
        return torch.abs(ifft_2d(x))
    return x


def _gaussian_kernel(size: int = 11, sigma: float = 1.5, device=None) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel = g[:, None] * g[None, :]
    kernel /= kernel.sum()
    return kernel.view(1, 1, size, size)


class MagnitudeImageLoss(nn.Module):
    """MSE in the normalised magnitude image domain."""
    def forward(self, pred, gt):
        return torch.mean((_to_magnitude(pred) - _to_magnitude(gt)) ** 2)


class MagnitudeL1Loss(nn.Module):
    """L1 loss in the normalised magnitude image domain."""
    def forward(self, pred, gt):
        return torch.mean(torch.abs(_to_magnitude(pred) - _to_magnitude(gt)))


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

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        p = _to_magnitude(pred)
        g = _to_magnitude(gt)
        return 1.0 - self._ssim(p, g)
