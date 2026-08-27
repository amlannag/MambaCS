import torch
from torch import nn
from .dc import KSpace_DC
from .vit import TokenVIT, axVIT, CrossAttentionVIT
from .encoders import TokenEncoder, axialEncoder, crossAxialEncoder

__all__ = ['cascadeNet', 'TokenVIT', 'axVIT', 'CrossAttentionVIT', 'TokenEncoder', 'axialEncoder', 'crossAxialEncoder']


class cascadeNet(nn.Module):
    """
    Cascaded denoising network with data consistency after each stage.

    Encoders operate in the configured normalized learning domain. Each candidate is
    restored to raw k-space for data consistency, then normalized back into that
    learning domain before the next cascade stage.

    Args:
        N (int)                 Image size
        encList (list)          Encoder classes for each cascade stage
        encArgs (list)          Dicts of kwargs for each encoder
        lamb (bool)             Whether to use a learned per-stage lambda
        learning (str)          "k_space", "image", or "complex_image"
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
        valid_domains = {"k_space", "image", "complex_image"}
        if learning not in valid_domains:
            raise ValueError(
                f"Unknown learning domain '{learning}'. Choose from: {sorted(valid_domains)}"
            )

        self.transformers = nn.ModuleList(
            enc(N, **args) for enc, args in zip(encList, encArgs)
        )

    def set_scheduled_lamb(self, value):
        self.scheduled_lamb = value

    def forward(self, xPrev, y, sampleMask, return_intermediates=False, stats=None):
        """
        xPrev      : [B,1,H,W] normalized model-domain input
        y          : [B,1,H,W] raw measured complex k-space
        sampleMask : [H, W]
        Returns same domain as xPrev. When return_intermediates=True, also returns
        the ordered list of post-DC stage states.
        """
        from normalizer import model_output_to_raw_kspace, raw_kspace_to_model_output

        x = xPrev
        intermediates = []
        for i, transformer in enumerate(self.transformers):
            if self.lamb is not False:
                lamb_i = self.lamb[i]
            elif self.scheduled_lamb is not None:
                lamb_i = self.scheduled_lamb
            else:
                lamb_i = None
            candidate = x + transformer(x, col_mask=sampleMask)
            raw_candidate_kspace = model_output_to_raw_kspace(
                candidate, stats, self.learning
            )
            raw_corrected_kspace = KSpace_DC(
                raw_candidate_kspace, y, sampleMask, lamb_i
            )
            x = raw_kspace_to_model_output(
                raw_corrected_kspace, stats, self.learning
            )
            if return_intermediates:
                intermediates.append(x)
        if return_intermediates:
            return x, intermediates
        return x
