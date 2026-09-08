import torch
from torch import nn
from .dc import KSpace_DC
from .vit import TokenVIT, axVIT, CrossAttentionVIT
from .encoders import TokenEncoder, axialEncoder, crossAxialEncoder, pair
from .util import FeedForward, _COMPLEX_ATTN_TYPES, validate_flattening_order

__all__ = ['cascadeNet', 'TokenVIT', 'axVIT', 'CrossAttentionVIT', 'TokenEncoder', 'axialEncoder', 'crossAxialEncoder']


def _stage_ffn_spec(N, cls, args):
    """Return (d_model, dim_feedforward, dropout, activation, is_complex) for one cascade stage."""
    num_ch = args.get("numCh", 1)
    if cls is TokenVIT:
        patch_h, patch_w = pair(args.get("patch_size", (16, 16)))
        d_model = args.get("d_model") or (patch_h * patch_w * num_ch)
    else:
        _, image_width = N if isinstance(N, (tuple, list)) else (N, N)
        d_model = args.get("d_model") or (image_width * num_ch)
    dim_ff = args.get("dim_feedforward") or int(d_model * 4)
    dropout = args.get("dropout", 0.1)
    activation = args.get("activation", "relu")
    is_complex = args.get("attn_type", "standard") in _COMPLEX_ATTN_TYPES
    return (d_model, dim_ff, dropout, activation, is_complex)


def _apply_ffn_sharing(N, encList, encArgs, ffn_sharing):
    """Return a copy of stage args with the FFN sharing mode applied."""
    if ffn_sharing == "none":
        return list(encArgs)
    if ffn_sharing == "per_stage":
        return [dict(args, ffn_sharing="per_stage") for args in encArgs]

    # global: one FeedForward shared by every stage
    specs = [_stage_ffn_spec(N, cls, args) for cls, args in zip(encList, encArgs)]
    base = specs[0]
    if any(spec != base for spec in specs[1:]):
        raise ValueError(
            "ffn_sharing='global' requires every cascade stage to use the same "
            "d_model, dim_feedforward, dropout, activation, and complex dtype. "
            f"Got stage specs: {specs}"
        )
    d_model, dim_ff, dropout, activation, is_complex = base
    shared_ffn = FeedForward(d_model, dim_ff, dropout, activation, is_complex)
    return [dict(args, shared_ffn=shared_ffn) for args in encArgs]


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
    def __init__(self, N, encList, encArgs, lamb=True, learning="k_space", ffn_sharing="none"):
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

        if ffn_sharing not in ("none", "per_stage", "global"):
            raise ValueError(
                f"Unknown ffn_sharing '{ffn_sharing}'. Choose from: none, per_stage, global"
            )
        self.ffn_sharing = ffn_sharing
        for args in encArgs:
            order = args.get('flattening_order', 'row_major')
            validate_flattening_order(order, args.get('tokenizer_type'))
            if order == 'dc_radial' and learning != 'k_space':
                raise ValueError("flattening_order='dc_radial' requires centered k-space (learning='k_space')")
        encArgs = _apply_ffn_sharing(N, encList, encArgs, ffn_sharing)

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
