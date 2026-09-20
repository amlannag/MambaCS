import torch
from torch import nn
from .dc import KSpace_DC
from .vit import TokenVIT, axVIT, CrossAttentionVIT, FNetVIT
from .fixed_apt import FixedAPTVIT
from .encoders import TokenEncoder, axialEncoder, crossAxialEncoder, pair
from .util import FeedForward, _COMPLEX_ATTN_TYPES, validate_flattening_order

__all__ = ['cascadeNet', 'TokenVIT', 'axVIT', 'CrossAttentionVIT', 'FNetVIT', 'FixedAPTVIT', 'TokenEncoder', 'axialEncoder', 'crossAxialEncoder']


def _stage_ffn_spec(N, cls, args):
    """Return (d_model, dim_feedforward, dropout, activation, is_complex) for one cascade stage."""
    num_ch = args.get("numCh", 1)
    if cls is FixedAPTVIT:
        d_model = args.get("d_model", 256)
    elif cls is TokenVIT:
        patch_h, patch_w = pair(args.get("patch_size", (16, 16)))
        d_model = args.get("d_model") or (patch_h * patch_w * num_ch)
    else:
        _, image_width = N if isinstance(N, (tuple, list)) else (N, N)
        d_model = args.get("d_model") or (image_width * num_ch)
    dim_ff = args.get("dim_feedforward") or int(d_model * 4)
    dropout = args.get("dropout", 0.1)
    activation = args.get("activation", "relu")
    default_attn_type = "complex" if cls is FixedAPTVIT else "standard"
    is_complex = args.get("attn_type", default_attn_type) in _COMPLEX_ATTN_TYPES
    return (d_model, dim_ff, dropout, activation, is_complex)


def _apply_ffn_sharing(N, encList, encArgs, ffn_sharing):
    """
    Return a copy of stage args with the FFN sharing mode applied.
    FNet stages are never part of FFN sharing: their per-token FFN is all they learn, and their hidden
    width may differ from the attention stages (un-embedded tokens), so they always keep their own FFNs.
    """
    if ffn_sharing == "none":
        return list(encArgs)
    shared = [cls is not FNetVIT for cls in encList]
    if ffn_sharing == "per_stage":
        return [dict(args, ffn_sharing="per_stage") if share else dict(args, ffn_sharing="none")
                for args, share in zip(encArgs, shared)]

    # global: one FeedForward shared by every non-FNet stage
    specs = [_stage_ffn_spec(N, cls, args) for cls, args, share in zip(encList, encArgs, shared) if share]
    if not specs:
        return [dict(args, ffn_sharing="none") for args in encArgs]
    base = specs[0]
    if any(spec != base for spec in specs[1:]):
        raise ValueError(
            "ffn_sharing='global' requires every cascade stage to use the same "
            "d_model, dim_feedforward, dropout, activation, and complex dtype. "
            f"Got stage specs: {specs}"
        )
    d_model, dim_ff, dropout, activation, is_complex = base
    shared_ffn = FeedForward(d_model, dim_ff, dropout, activation, is_complex)
    return [dict(args, shared_ffn=shared_ffn) if share else dict(args, ffn_sharing="none")
            for args, share in zip(encArgs, shared)]


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
        if FixedAPTVIT in encList and learning != 'k_space':
            raise ValueError("fixed_apt requires learning='k_space'")
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

        Data consistency runs directly in the model's normalized domain: the raw
        measured k-space is converted into the normalized domain once (using the
        same transform as the input), and each stage blends the candidate with it
        in place. Nothing is converted back to raw units inside the cascade; that
        happens only at inference/validation via model_output_to_raw_kspace.
        For k-space learning with k-space-domain normalization (robust_shifted,
        kspace_companding, log_kspace, fastmri_magnitude) the transform and the
        DC both stay in k-space with no image-domain round trip. The raw-domain
        path is retained for the real-image learning mode or missing stats.
        """
        from normalizer import apply_normalization, model_output_to_raw_kspace, raw_kspace_to_model_output
        from DcTNN.dc import fft_2d, ifft_2d

        use_normalized_dc = stats is not None and self.learning in ("k_space", "complex_image")
        normalization_domain = stats.get("normalization_domain", self.learning) if stats else self.learning
        # k-space learning always operates on k-space tensors, so DC blends the
        # candidate's k-space coefficients directly. complex_image operates on
        # images, so the blend happens in k-space via one FFT/IFFT pair.
        dc_domain_is_k_space = use_normalized_dc and self.learning == "k_space"
        y_model = None
        if use_normalized_dc:
            if normalization_domain == "k_space":
                # K-space-domain normalization (robust_shifted, kspace_companding,
                # log_kspace, fastmri_magnitude): normalize the measured k-space
                # directly, no image-domain detour at all.
                y_model = apply_normalization(y, stats)
            else:
                # Image-domain normalization (zscore, complex_image): normalize the
                # measured k-space in the model's image domain and FFT once so DC
                # still blends k-space coefficients without denormalizing.
                y_model = fft_2d(apply_normalization(ifft_2d(y), stats))

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
            if use_normalized_dc:
                candidate_kspace = candidate if dc_domain_is_k_space else fft_2d(candidate)
                if lamb_i is None:
                    corrected = (1 - sampleMask) * candidate_kspace + sampleMask * y_model
                else:
                    corrected = (1 - sampleMask) * candidate_kspace + sampleMask * (
                        candidate_kspace + lamb_i * y_model
                    ) / (1 + lamb_i)
                x = corrected if dc_domain_is_k_space else ifft_2d(corrected)
            else:
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
