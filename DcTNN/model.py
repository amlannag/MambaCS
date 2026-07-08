import torch
from torch import nn
from .dc import FFT_DC, KSpace_DC
from .vit import TokenVIT, axVIT, CrossAttentionVIT
from .encoders import TokenEncoder, axialEncoder, crossAxialEncoder

__all__ = ['cascadeNet', 'TokenVIT', 'axVIT', 'CrossAttentionVIT', 'TokenEncoder', 'axialEncoder', 'crossAxialEncoder']


class cascadeNet(nn.Module):
    """
    Cascaded denoising network with data consistency after each stage.

    When learning="k_space" — encoder input/output is complex k-space; KSpace_DC used.
    When learning="image"   — encoder input/output is real magnitude image; FFT_DC used.

    Args:
        N (int)                 Image size
        encList (list)          Encoder classes for each cascade stage
        encArgs (list)          Dicts of kwargs for each encoder
        lamb (bool)             Whether to use a learned per-stage lambda
        learning (str)          "k_space" or "image"
    """
    def __init__(self, N, encList, encArgs, lamb=True, learning="k_space"):
        super().__init__()
        if lamb:
            self.lamb = nn.Parameter(torch.ones(len(encList)) * 0.5)
        else:
            self.lamb = False
        self.scheduled_lamb = None
        self.N = N
        self.learning = learning
        self._dc_func = KSpace_DC if learning == "k_space" else FFT_DC

        self.transformers = nn.ModuleList(
            enc(N, **args) for enc, args in zip(encList, encArgs)
        )

    def set_scheduled_lamb(self, value):
        self.scheduled_lamb = value

    def forward(self, xPrev, y, sampleMask, return_intermediates=False):
        """
        xPrev      : [B,1,H,W] complex undersampled k-space  (learning="k_space")
                     [B,1,H,W] real magnitude undersampled    (learning="image")
        y          : [B,1,H,W] complex k-space DC reference (always complex)
        sampleMask : [H, W]
        Returns same domain as xPrev. When return_intermediates=True, also returns
        the ordered list of post-DC stage states.
        """
        x = xPrev
        intermediates = []
        for i, transformer in enumerate(self.transformers):
            
            if self.lamb is not False:
                lamb_i = self.lamb[i]
            elif self.scheduled_lamb is not None:
                lamb_i = self.scheduled_lamb
            else:
                lamb_i = None
            x = self._dc_func(x + transformer(x, col_mask=sampleMask), y, sampleMask, lamb_i)
            if self.learning == "image":
                x = x.real    # FFT_DC IFFTs to complex; extract real for next encoder
            if return_intermediates:
                intermediates.append(x)
        if return_intermediates:
            return x, intermediates
        return x
