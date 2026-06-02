import torch
from torch import nn
from .dc import FFT_DC, KSpace_DC
from .vit import TokenVIT, axVIT
from .encoders import TokenEncoder, axialEncoder

__all__ = ['cascadeNet', 'TokenVIT', 'axVIT', 'TokenEncoder', 'axialEncoder']


class cascadeNet(nn.Module):
    """
    Cascaded denoising network with data consistency after each stage.

    When k_space_learning=True  — input/output are k-space [B, 1, N, N] complex;
                                   KSpace_DC enforces consistency in k-space.
    When k_space_learning=False — input/output are image domain [B, 1, N, N] complex;
                                   FFT_DC converts to k-space internally for DC.

    Args:
        N (int)                 Image size
        encList (list)          Encoder classes for each cascade stage
        encArgs (list)          Dicts of kwargs for each encoder
        lamb (bool)             Whether to use a learned per-stage lambda
        k_space_learning (bool) True → k-space mode (default); False → image mode
    """
    def __init__(self, N, encList, encArgs, lamb=True, k_space_learning=True):
        super().__init__()
        if lamb:
            self.lamb = nn.Parameter(torch.ones(len(encList)) * 0.5)
        else:
            self.lamb = False
        self.scheduled_lamb = None
        self.N = N
        self.k_space_learning = k_space_learning
        self._dc_func = KSpace_DC if k_space_learning else FFT_DC

        self.transformers = nn.ModuleList(
            enc(N, **args) for enc, args in zip(encList, encArgs)
        )

    def set_scheduled_lamb(self, value):
        self.scheduled_lamb = value

    def forward(self, xPrev, y, sampleMask):
        """
        xPrev      : [B, 1, N, N] complex undersampled k-space (k_space_learning=True)
                     [B, 1, N, N] complex zero-filled image    (k_space_learning=False)
        y          : [B, 1, N, N] complex undersampled k-space (DC reference, always)
        sampleMask : [N, N]
        Returns same domain as xPrev.
        """
        x = xPrev
        for i, transformer in enumerate(self.transformers):
            
            if self.lamb is not False:
                lamb_i = self.lamb[i]
            elif self.scheduled_lamb is not None:
                lamb_i = self.scheduled_lamb
            else:
                lamb_i = None
            x = self._dc_func(x + transformer(x), y, sampleMask, lamb_i)
        return x
