"""Loss functions for MambaCS reconstruction."""
import torch
from torch import nn
from DcTNN.dc import ifft_2d


def _to_magnitude(x):
    """
    Bring x into the real magnitude image domain.
    - complex (k-space): ifft2 → abs → real magnitude, values ≥ 0
    - real (image domain): pass through as-is, values in [0, 1] after min-max normalisation
    """
    if x.is_complex():
        return torch.abs(ifft_2d(x))
    return x


class MagnitudeImageLoss(nn.Module):
    """
    MSE in the normalised magnitude image domain.

    Pipeline per input:
      - complex k-space [B,1,H,W] cfloat → ifft2 → abs → real magnitude
      - real image      [B,1,H,W] float  → used directly (normalised, may be negative)

    pred (model output) and gt (gt_norm from simulate_undersampling) need not be
    in the same domain — each is converted independently.
    """
    def forward(self, pred, gt):
        return torch.mean((_to_magnitude(pred) - _to_magnitude(gt)) ** 2)
