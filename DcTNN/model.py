import math

import torch
from torch import nn
import torch.nn.functional as F
from .dc import KSpace_DC
from .vit import TokenVIT, axVIT, CrossAttentionVIT, PerCoilCrossAttentionVIT, NeighbourhoodVIT, FNetVIT, PCAVIT
from .fixed_apt import FixedAPTVIT
from .band_split import BandSplitVIT, FreqSplitSigmas, gaussian_partition, normalized_radius_grid
from .encoders import TokenEncoder, axialEncoder, crossAxialEncoder, pair
from .util import FeedForward, _COMPLEX_ATTN_TYPES, validate_flattening_order

__all__ = ['cascadeNet', 'TokenVIT', 'axVIT', 'CrossAttentionVIT', 'FNetVIT', 'FixedAPTVIT', 'BandSplitVIT', 'TokenEncoder', 'axialEncoder', 'crossAxialEncoder']


def _flatten_stages(encList, encArgs):
    """
    Leaf (cls, args) pairs of a cascade: a BandSplitVIT stage contributes its two experts (args["low"], args["high"],
    each a (cls, args) pair) instead of itself. Returns (leaf classes, leaf args, layout) where layout[i] is None for a
    plain stage or (low_leaf_index, high_leaf_index) for a split stage, so `_unflatten_stages` can rebuild the lists.
    """
    leaf_cls, leaf_args, layout = [], [], []
    for cls, args in zip(encList, encArgs):
        if cls is BandSplitVIT:
            layout.append((len(leaf_cls), len(leaf_cls) + 1))
            for key in ("low", "high"):
                sub_cls, sub_args = args[key]
                leaf_cls.append(sub_cls)
                leaf_args.append(sub_args)
        else:
            layout.append(len(leaf_cls))
            leaf_cls.append(cls)
            leaf_args.append(args)
    return leaf_cls, leaf_args, layout


def _unflatten_stages(encArgs, leaf_cls, leaf_args, layout):
    """Inverse of _flatten_stages: stage args with (possibly updated) leaf args put back in place."""
    out = []
    for args, slot in zip(encArgs, layout):
        if isinstance(slot, tuple):
            low, high = slot
            out.append(dict(args, low=(leaf_cls[low], leaf_args[low]), high=(leaf_cls[high], leaf_args[high])))
        else:
            out.append(leaf_args[slot])
    return out


def _stage_ffn_spec(N, cls, args):
    """Return (d_model, dim_feedforward, dropout, activation, is_complex) for one cascade stage."""
    num_ch = args.get("numCh", 1)
    if cls is FixedAPTVIT or (cls is PCAVIT and args.get("tokenizer") == "fixed_apt"):
        d_model = args.get("d_model") or args.get("apt_embed_dim", 256)
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
    FNet stages are excluded from FFN sharing by default: their per-token FFN is all they learn, and their hidden
    width may differ from the attention stages (un-embedded tokens), so they keep their own FFNs unless the stage
    args set `share_ffn=True` (requires embedded tokens so the width matches the attention stages).
    The experts of a BandSplitVIT stage take part individually (an axial expert shares with the other axial stages, ...).
    """
    leaf_cls, leaf_args, layout = _flatten_stages(encList, encArgs)
    leaf_args = _apply_ffn_sharing_flat(N, leaf_cls, leaf_args, ffn_sharing)
    return _unflatten_stages(encArgs, leaf_cls, leaf_args, layout)


def _apply_ffn_sharing_flat(N, encList, encArgs, ffn_sharing):
    encArgs = [dict(args) for args in encArgs]
    share_flags = [args.pop("share_ffn", False) for args in encArgs]
    if ffn_sharing == "none":
        return encArgs
    shared = [cls is not FNetVIT or flag for cls, flag in zip(encList, share_flags)]
    if ffn_sharing == "per_stage":
        return [dict(args, ffn_sharing="per_stage") if share else dict(args, ffn_sharing="none")
                for args, share in zip(encArgs, shared)]

    # global: one FeedForward shared by every non-FNet stage of the same encoder type (axial with axial,
    # fixed_apt with fixed_apt, ...); stages of different types never share.
    specs = {}
    for cls, args, share in zip(encList, encArgs, shared):
        if share:
            specs.setdefault(cls, []).append(_stage_ffn_spec(N, cls, args))
    shared_ffns = {}
    for cls, cls_specs in specs.items():
        base = cls_specs[0]
        if any(spec != base for spec in cls_specs[1:]):
            raise ValueError(
                f"ffn_sharing='global' requires every {cls.__name__} cascade stage to use the same "
                "d_model, dim_feedforward, dropout, activation, and complex dtype. "
                f"Got stage specs: {cls_specs}"
            )
        shared_ffns[cls] = FeedForward(*base)
    return [dict(args, shared_ffn=shared_ffns[cls]) if share else dict(args, ffn_sharing="none")
            for cls, args, share in zip(encList, encArgs, shared)]


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
        dc_type (str)           "scalar": one learned lambda per stage (default);
                                "pixelwise": one learned lambda per k-space pixel and stage ([stages, H, W],
                                softplus-constrained > 0, so it works with any sampling mask incl. random ones;
                                only the sampled pixels of a given mask have an effect)
        dc_lambda_init (float)  Initial value of every lambda (0.5 for the original scalar DC)
        learning (str)          "k_space", "image", or "complex_image"
        real_channels (bool)    k_space only: run the whole cascade (encoders and data consistency) on a
                                real [B, 2, H, W] tensor whose channels are the real and imaginary parts
                                of the complex k-space, instead of a complex [B, 1, H, W] tensor. Encoders
                                must then be real-valued with numCh doubled. The complex k-space is only
                                recovered at the cascade output so the loss/validation path is unchanged.
        freq_split_sigma_init / _learned / _bounds
                                sigma of every BandSplitVIT stage in cascade order (DcTNN/band_split.py): the
                                Gaussian partition boundaries, non-decreasing along the cascade, learned or fixed.
                                A BandSplitVIT stage's args hold its experts as args["low"] / args["high"] = (cls, args).
    """
    def __init__(self, N, encList, encArgs, lamb=True, learning="k_space", ffn_sharing="none", real_channels=False,
                 dc_type="scalar", dc_lambda_init=0.5, freq_split_sigma_init=None, freq_split_sigma_learned=True,
                 freq_split_sigma_bounds=(0.05, math.sqrt(2.0))):
        super().__init__()
        self.real_channels = bool(real_channels)
        self.split_stage_index = {i: n for n, i in enumerate(i for i, cls in enumerate(encList) if cls is BandSplitVIT)}
        self.freq_split_sigmas = None
        if self.split_stage_index:
            if learning != "k_space":
                raise ValueError("freq_split stages require learning='k_space'")
            if self.real_channels:
                raise ValueError("freq_split stages cannot be used with real_channels")
            if freq_split_sigma_init is None or len(freq_split_sigma_init) != len(self.split_stage_index):
                raise ValueError(f"freq_split_sigma_init needs one value per freq_split stage ({len(self.split_stage_index)})")
            self.freq_split_sigmas = FreqSplitSigmas(freq_split_sigma_init, learned=freq_split_sigma_learned,
                                                     bounds=freq_split_sigma_bounds)
        leaf_cls, leaf_args, _ = _flatten_stages(encList, encArgs)
        if self.real_channels:
            if learning != "k_space":
                raise ValueError("real_channels requires learning='k_space'")
            for cls, args in zip(leaf_cls, leaf_args):
                if cls in (FixedAPTVIT, CrossAttentionVIT, PerCoilCrossAttentionVIT, NeighbourhoodVIT, PCAVIT):
                    raise ValueError(f"{cls.__name__} is complex-only and cannot be used with real_channels")
                if args.get("attn_type", "standard") in _COMPLEX_ATTN_TYPES:
                    raise ValueError("real_channels requires a real-valued attn_type ('standard') on every stage")
                if args.get("numCh", 1) % 2:
                    raise ValueError("real_channels needs an even numCh (2 real channels per complex channel)")
        if dc_type not in ("scalar", "pixelwise"):
            raise ValueError(f"Unknown dc_type '{dc_type}'. Choose from: scalar, pixelwise")
        self.dc_type = dc_type
        self.dc_weights = None
        if lamb and dc_type == "pixelwise":
            height, width = N if isinstance(N, (tuple, list)) else (N, N)
            # raw weights; lambda = softplus(raw) so every pixel's lambda stays positive, initialised at dc_lambda_init
            self.dc_weights = nn.Parameter(torch.full((len(encList), height, width), math.log(math.expm1(dc_lambda_init))))
            self.lamb = False
        elif lamb:
            self.lamb = nn.Parameter(torch.ones(len(encList)) * dc_lambda_init)
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
        if FixedAPTVIT in leaf_cls and learning != 'k_space':
            raise ValueError("fixed_apt requires learning='k_space'")
        for args in leaf_args:
            order = args.get('flattening_order', 'row_major')
            validate_flattening_order(order, args.get('tokenizer_type'))
            if order == 'dc_radial' and learning != 'k_space':
                raise ValueError("flattening_order='dc_radial' requires centered k-space (learning='k_space')")
        encArgs = _apply_ffn_sharing(N, encList, encArgs, ffn_sharing)

        self.transformers = nn.ModuleList(
            BandSplitVIT(N, args["low"][0](N, **args["low"][1]), args["high"][0](N, **args["high"][1]))
            if enc is BandSplitVIT else enc(N, **args)
            for enc, args in zip(encList, encArgs)
        )

    def set_scheduled_lamb(self, value):
        self.scheduled_lamb = value

    def freq_split_sigma_values(self):
        """{stage index: sigma} of the BandSplitVIT stages (empty when the cascade has none)."""
        if self.freq_split_sigmas is None:
            return {}
        values = self.freq_split_sigmas.values()
        return {stage: values[n] for stage, n in self.split_stage_index.items()}

    def stage_band_weights(self, band="low", floor=0.0, sigmas=None, normalise="mean"):
        """
        Per-stage radial loss weights for loss_mode='intermediate_banded': one [1, 1, H, W] map per stage.

        With explicit `sigmas` (one per stage, independent of the gates) every stage i is weighted by the disc
        G(r; sigma_i) = exp(-(r/sigma_i)^2/2) (band='low': the stage answers for frequencies inside sigma_i; a growing
        schedule gives low -> high supervision along the cascade), by 1 - G (band='high'), or by the ANNULUS the stage
        is newly responsible for (band='ring': w_1 = G(r; s_1), w_i = G(r; s_i) - G(r; s_{i-1}), and the LAST stage gets
        everything outside the previous disc, w_n = 1 - G(r; s_{n-1}), so the cascade ends by prioritising the high
        frequencies instead of merely including them; needs explicit sigmas). Without `sigmas` the freq_split gates
        themselves are used (detached) and non-split stages get None (plain loss). Weights are lifted to at least
        `floor` and then normalised: "mean" -> mean 1 over the grid (every stage gets an equal vote regardless of its
        band's area: the curriculum re-weights the spectrum), "none" -> raw values (ring bands form a partition of
        unity, so the stage terms are an unbiased split of the plain loss; no cell counted twice), "max" -> peak 1.
        """
        if band not in ("high", "low", "ring"):
            raise ValueError(f"band must be 'high', 'low' or 'ring', got {band!r}")
        if normalise not in ("mean", "none", "max"):
            raise ValueError(f"normalise must be 'mean', 'none' or 'max', got {normalise!r}")
        n = len(self.transformers)
        radius = normalized_radius_grid(self.N, device=next(self.parameters()).device)
        if sigmas is not None:
            if len(sigmas) != n:
                raise ValueError(f"banded loss needs one sigma per stage ({n}), got {len(sigmas)}")
            gates = [gaussian_partition(radius, float(s)).unsqueeze(0).unsqueeze(0) for s in sigmas]
        else:
            if band == "ring":
                raise ValueError("band='ring' needs explicit banded loss sigmas (banded_loss_sigma_start/end or banded_loss_sigmas)")
            if self.freq_split_sigmas is None:
                return [None] * n
            learned = self.freq_split_sigmas().detach()
            gates = [
                transformer.partition(learned[self.split_stage_index[i]]) if isinstance(transformer, BandSplitVIT) else None
                for i, transformer in enumerate(self.transformers)
            ]
        weights = []
        for i, g in enumerate(gates):
            if g is None:
                weights.append(None)
                continue
            if band == "ring":
                inner = gates[i - 1] if i > 0 else torch.zeros_like(g)
                w = (1.0 - inner) if i == n - 1 and n > 1 else (g - inner)
            else:
                w = (1.0 - g) if band == "high" else g
            w = w.clamp_min(floor)
            if normalise == "mean":
                w = w / w.mean().clamp_min(1e-12)
            elif normalise == "max":
                w = w / w.max().clamp_min(1e-12)
            weights.append(w)
        return weights

    def _stage_residual(self, i, transformer, x, sampleMask, volume_id=None):
        if isinstance(transformer, PCAVIT):
            return transformer(x, col_mask=sampleMask, volume_id=volume_id)
        if isinstance(transformer, BandSplitVIT):
            return transformer(x, col_mask=sampleMask, sigma=self.freq_split_sigmas()[self.split_stage_index[i]])
        return transformer(x, col_mask=sampleMask)

    def forward(self, xPrev, y, sampleMask, return_intermediates=False, stats=None, volume_id=None):
        """
        xPrev      : [B,1,H,W] normalized model-domain input
        volume_id  : optional [B] ids grouping slices into volumes (used by PCA stages)
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

        if self.real_channels:
            return self._forward_real_channels(xPrev, y, sampleMask, return_intermediates, stats)

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
            lamb_i = self._lamb_for_stage(i)
            with torch.profiler.record_function(f"stage{i + 1}:{type(transformer).__name__}"):
                candidate = x + self._stage_residual(i, transformer, x, sampleMask, volume_id)
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

    def _lamb_for_stage(self, i):
        """Stage lambda: scalar parameter, [H, W] pixelwise map (broadcast over batch/channels), schedule value, or None."""
        if self.dc_weights is not None:
            return F.softplus(self.dc_weights[i])
        if self.lamb is not False:
            return self.lamb[i]
        return self.scheduled_lamb

    def lambda_map(self, i):
        """[H, W] lambda map of stage i (scalar lambdas expanded), for inspection."""
        lamb = self._lamb_for_stage(i)
        height, width = self.N if isinstance(self.N, (tuple, list)) else (self.N, self.N)
        if lamb is None:
            return None
        return torch.as_tensor(lamb).expand(height, width) if torch.as_tensor(lamb).ndim == 0 else lamb

    def _forward_real_channels(self, xPrev, y, sampleMask, return_intermediates, stats):
        """
        k_space cascade on a real [B, 2C, H, W] (re, im) tensor. The complex input and measured k-space are
        split once on entry; every stage and every DC blend then works on real channels, and the complex
        k-space is rebuilt only for the outputs. DC is per-element and linear, so blending the two real
        channels separately is exactly the complex blend.
        """
        from normalizer import apply_normalization
        from DcTNN.dc import fft_2d, ifft_2d, complex_to_pair, pair_to_complex

        normalization_domain = stats.get("normalization_domain", "k_space") if stats else "k_space"
        if stats is None or normalization_domain == "k_space":
            y_model = apply_normalization(y, stats)
        else:
            y_model = fft_2d(apply_normalization(ifft_2d(y), stats))
        y_pair = complex_to_pair(y_model)

        x = complex_to_pair(xPrev)
        intermediates = []
        for i, transformer in enumerate(self.transformers):
            lamb_i = self._lamb_for_stage(i)
            candidate = x + transformer(x, col_mask=sampleMask)
            if lamb_i is None:
                x = (1 - sampleMask) * candidate + sampleMask * y_pair
            else:
                x = (1 - sampleMask) * candidate + sampleMask * (candidate + lamb_i * y_pair) / (1 + lamb_i)
            if return_intermediates:
                intermediates.append(pair_to_complex(x))
        out = pair_to_complex(x)
        if return_intermediates:
            return out, intermediates
        return out
