"""
Shared helpers used by training and inference.
"""

import torch
from fastmri.data.subsample import EquiSpacedMaskFunc, RandomMaskFunc
from fastmri.data.transforms import apply_mask

from DcTNN.model import TokenVIT, axVIT, CrossAttentionVIT, FixedAPTVIT, cascadeNet
from DcTNN.fixed_apt import resolve_fixed_apt_layout
from DcTNN.util import _COMPLEX_ATTN_TYPES
from ReconFormer import ReconFormerBaseline
import normalizer as _norm


DATASET_DIRS = {
    "fastmri": (
        "/scratch/user/uqanag/fastmri/singlecoil_train",
        "/scratch/user/uqanag/fastmri/singlecoil_val",
    ),
    "oasis": (
        "/scratch/user/uqanag/OASIS/keras_png_slices_train",
        "/scratch/user/uqanag/OASIS/keras_png_slices_validate",
    ),
}


def resolve_data_dirs(cfg):
    """Return (train_dir, val_dir). Falls back to DATASET_DIRS[cfg.dataset] if not set."""
    default_train, default_val = DATASET_DIRS.get(cfg.dataset, (None, None))
    train_dir = cfg.data_dir    or default_train
    val_dir   = cfg.val_data_dir or default_val or train_dir
    if train_dir is None:
        raise ValueError(
            f"No data_dir for dataset='{cfg.dataset}'. "
            f"Set data_dir in train_config.py or add an entry to DATASET_DIRS."
        )
    return train_dir, val_dir


def psnr(pred, target, max_val=None):
    """PSNR between pred and target. max_val defaults to target.max()."""
    mse = torch.mean(torch.abs(pred - target) ** 2)
    if mse == 0:
        return torch.tensor(float("inf"))
    mv = target.max() if max_val is None else torch.tensor(max_val, device=pred.device)
    return 20.0 * torch.log10(mv.to(pred.device) / torch.sqrt(mse))


_ENCODER_ARGS = {
    "fixed_apt": lambda cfg: (
        FixedAPTVIT,
        dict(
            layerNo=cfg.layer_no,
            numCh=cfg.num_channels,
            d_model=getattr(cfg, "apt_embed_dim", 256),
            nhead=cfg.nhead_patch,
            num_encoder_layers=cfg.num_encoder_layers,
            layer_norm_eps=getattr(cfg, "layer_norm_eps", 1e-5),
            dim_feedforward=None,
            attn_type=cfg.attn_type,
            layout=resolve_fixed_apt_layout(getattr(cfg, "apt_layout", None)),
            rope_theta=cfg.rope_theta,
            rope_ref_grid=getattr(cfg, "apt_rope_ref_grid", None),
            use_abs_pos_emb=getattr(cfg, "apt_use_abs_pos_emb", False),
        ),
    ),
    "axial": lambda cfg: (
        axVIT,
        dict(
            layerNo=cfg.layer_no,
            numCh=cfg.num_channels,
            d_model=None,
            nhead=cfg.nhead_axial,
            num_encoder_layers=cfg.num_encoder_layers,
            layer_norm_eps=getattr(cfg, "layer_norm_eps", 1e-5),
            dim_feedforward=None,
            pos_emb_type=cfg.pos_emb_type,
            rope_theta=cfg.rope_theta,
            rope_mixed_rotate=cfg.rope_mixed_rotate,
            attn_type=cfg.attn_type,
            row_stride=cfg.axial_row_stride,
            mask_vertical_attn=cfg.mask_vertical_attn,
        ),
    ),
    "cross_axial": lambda cfg: (
        CrossAttentionVIT,
        dict(
            layerNo=cfg.layer_no,
            numCh=cfg.num_channels,
            d_model=None,
            nhead=cfg.nhead_axial,
            num_encoder_layers=cfg.num_encoder_layers,
            layer_norm_eps=getattr(cfg, "layer_norm_eps", 1e-5),
            dim_feedforward=None,
            pos_emb_type=cfg.pos_emb_type,
            rope_theta=cfg.rope_theta,
            rope_mixed_rotate=cfg.rope_mixed_rotate,
            attn_type=cfg.attn_type,
            row_stride=cfg.axial_row_stride,
        ),
    ),
    "kaleidoscope": lambda cfg: (
        TokenVIT,
        dict(
            patch_size=cfg.patch_size,
            tokenizer_type="kaleidoscope",
            layerNo=cfg.layer_no,
            numCh=cfg.num_channels,
            nhead=cfg.nhead_patch,
            num_encoder_layers=cfg.num_encoder_layers,
            layer_norm_eps=getattr(cfg, "layer_norm_eps", 1e-5),
            dim_feedforward=None,
            d_model=None,
            pos_emb_type=cfg.pos_emb_type,
            rope_theta=cfg.rope_theta,
            rope_mixed_rotate=cfg.rope_mixed_rotate,
            attn_type=cfg.attn_type,
        ),
    ),
    "patch": lambda cfg: (
        TokenVIT,
        dict(
            patch_size=cfg.patch_size,
            tokenizer_type="patch",
            layerNo=cfg.layer_no,
            numCh=cfg.num_channels,
            nhead=cfg.nhead_patch,
            num_encoder_layers=cfg.num_encoder_layers,
            layer_norm_eps=getattr(cfg, "layer_norm_eps", 1e-5),
            dim_feedforward=None,
            d_model=None,
            pos_emb_type=cfg.pos_emb_type,
            rope_theta=cfg.rope_theta,
            rope_mixed_rotate=cfg.rope_mixed_rotate,
            attn_type=cfg.attn_type,
        ),
    ),
}


def validate_resume_flattening_order(cfg, saved_config):
    saved_model = saved_config.get("model", saved_config)
    saved_order = saved_model.get("flattening_order", "row_major")
    requested_order = getattr(cfg, "flattening_order", "row_major")
    if saved_order != requested_order:
        raise ValueError(
            f"Checkpoint flattening_order={saved_order!r} does not match "
            f"requested flattening_order={requested_order!r}; resume with the saved order."
        )


def validate_resume_fixed_apt_config(cfg, saved_config):
    saved_model = saved_config.get("model", saved_config)
    saved_data = saved_config.get("data", saved_config)
    saved_encoders = saved_model.get("encoders", [])
    requested_encoders = getattr(cfg, "encoders", [])
    if "fixed_apt" not in saved_encoders and "fixed_apt" not in requested_encoders:
        return
    validate_resume_flattening_order(cfg, saved_config)
    if list(saved_encoders) != list(requested_encoders):
        raise ValueError("Checkpoint fixed_apt encoders do not match requested encoders")
    if not isinstance(saved_model.get("apt_layout"), dict):
        raise ValueError("Checkpoint fixed_apt config must contain an explicit apt_layout dictionary")
    saved_layout = resolve_fixed_apt_layout(saved_model["apt_layout"])
    requested_layout = resolve_fixed_apt_layout(getattr(cfg, "apt_layout", None))
    for key in ("version", "image_size", "base_patch_size", "leaves"):
        if saved_layout[key] != requested_layout[key]:
            raise ValueError(f"Checkpoint fixed_apt apt_layout {key} does not match requested layout")
    defaults = {
        "apt_embed_dim": 256,
        "apt_use_abs_pos_emb": False,
        "rope_theta": 100.0,
        "nhead_patch": 8,
        "pos_emb_type": "APE",
        "attn_type": "standard",
        "learning": "k_space",
        "layer_no": 1,
        "num_encoder_layers": 2,
        "ffn_sharing": "none",
        "model_type": "dctnn",
    }
    for key, default in defaults.items():
        saved_value = saved_model.get(key, default)
        requested_value = getattr(cfg, key, default)
        if saved_value != requested_value:
            raise ValueError(
                f"Checkpoint fixed_apt {key}={saved_value!r} does not match "
                f"requested {key}={requested_value!r}"
            )
    saved_ref = saved_model.get("apt_rope_ref_grid")
    requested_ref = getattr(cfg, "apt_rope_ref_grid", None)
    default_ref = tuple(size // saved_layout["base_patch_size"] for size in saved_layout["image_size"])
    saved_ref = default_ref if saved_ref is None else tuple(saved_ref)
    requested_ref = default_ref if requested_ref is None else tuple(requested_ref)
    if saved_ref != requested_ref:
        raise ValueError("Checkpoint fixed_apt apt_rope_ref_grid does not match requested apt_rope_ref_grid")
    saved_size = saved_data.get("image_size", saved_layout["image_size"])
    saved_size = (saved_size, saved_size) if isinstance(saved_size, int) else tuple(saved_size)
    requested_size = (cfg.image_size, cfg.image_size) if isinstance(cfg.image_size, int) else tuple(cfg.image_size)
    if saved_size != requested_size:
        raise ValueError("Checkpoint fixed_apt image_size does not match requested image_size")
    if saved_data.get("num_channels", 1) != cfg.num_channels:
        raise ValueError("Checkpoint fixed_apt num_channels does not match requested num_channels")


def _build_model_impl(cfg):
    flattening_order = getattr(cfg, "flattening_order", "row_major")
    if "fixed_apt" in cfg.encoders:
        if cfg.model_type != "dctnn":
            raise ValueError("fixed_apt requires model_type='dctnn'")
        if cfg.learning != "k_space":
            raise ValueError("fixed_apt requires learning='k_space'")
        if flattening_order != "row_major":
            raise ValueError("fixed_apt requires flattening_order='row_major'")
        if cfg.pos_emb_type != "Rope-Axial":
            raise ValueError("fixed_apt requires pos_emb_type='Rope-Axial'")
        if cfg.attn_type not in _COMPLEX_ATTN_TYPES:
            raise ValueError(
                "fixed_apt requires a native complex attention type; "
                f"choose from {sorted(_COMPLEX_ATTN_TYPES)}"
            )
    if flattening_order not in ("row_major", "dc_radial"):
        raise ValueError(
            f"Unknown flattening_order {flattening_order!r}. Choose from: row_major, dc_radial"
        )
    if flattening_order == "dc_radial":
        if cfg.model_type != "dctnn":
            raise ValueError("flattening_order='dc_radial' requires model_type='dctnn'")
        if cfg.learning != "k_space":
            raise ValueError("flattening_order='dc_radial' requires learning='k_space'")
        if "kaleidoscope" in cfg.encoders:
            raise ValueError("flattening_order='dc_radial' is not supported for kaleidoscope")
    if cfg.model_type == "reconformer":
        if cfg.learning not in {"complex_image", "k_space"}:
            raise ValueError("ReconFormer requires learning='complex_image' or 'k_space'")
        if cfg.learning == "k_space" and cfg.norm == "reconformer":
            raise ValueError("ReconFormer in k-space needs a k-space-compatible norm (e.g. fastmri_magnitude), not 'reconformer'")
        image_size = tuple(int(value) for value in cfg.image_size)
        if len(image_size) != 2 or image_size[0] != image_size[1]:
            raise ValueError(f"ReconFormer requires a square image_size, got {image_size}")
        return ReconFormerBaseline(
            domain="kspace" if cfg.learning == "k_space" else "image",
            num_ch=cfg.reconformer_num_ch,
            down_scales=cfg.reconformer_down_scales,
            num_iter=cfg.reconformer_num_iter,
            img_size=image_size[0],
            num_heads=cfg.reconformer_num_heads,
            depths=cfg.reconformer_depths,
            window_sizes=cfg.reconformer_window_sizes,
            resi_connection=cfg.reconformer_resi_connection,
            mlp_ratio=cfg.reconformer_mlp_ratio,
            use_checkpoint=cfg.reconformer_use_checkpoint,
        )
    if cfg.model_type != "dctnn":
        raise ValueError(f"Unknown model_type {cfg.model_type!r}")

    num_ch = cfg.num_channels
    if cfg.learning == "complex_image" and cfg.attn_type == "standard":
        raise ValueError("learning='complex_image' requires a complex-valued attention type")

    stage_layers = getattr(cfg, "stage_encoder_layers", None)
    if stage_layers is not None and len(stage_layers) != len(cfg.encoders):
        raise ValueError(
            f"stage_encoder_layers has {len(stage_layers)} entries but encoders has {len(cfg.encoders)}"
        )

    enc_list = []
    enc_args = []
    for i, name in enumerate(cfg.encoders):
        if name not in _ENCODER_ARGS:
            raise ValueError(f"Unknown encoder '{name}'. Choose from: {list(_ENCODER_ARGS)}")
        cls, args = _ENCODER_ARGS[name](cfg)
        args["numCh"] = num_ch
        args["flattening_order"] = flattening_order
        if stage_layers is not None and "num_encoder_layers" in args:
            args["num_encoder_layers"] = int(stage_layers[i])
        enc_list.append(cls)
        enc_args.append(args)

    use_learned_lamb = cfg.lambda_schedule == "none"
    return cascadeNet(
        cfg.image_size,
        enc_list,
        enc_args,
        use_learned_lamb,
        learning=cfg.learning,
        ffn_sharing=cfg.ffn_sharing,
    )


def build_model(cfg):
    return _build_model_impl(cfg)


def unique_model_parameters(model):
    """
    Yield trainable parameters once each, deduplicating tied/shared weights.

    With ffn_sharing != "none" the same FeedForward module is registered under
    several parents, so model.parameters() yields the shared tensors multiple
    times. Optimizers and parameter counts must use this instead.
    """
    seen = set()
    for p in model.parameters():
        if p.requires_grad and id(p) not in seen:
            seen.add(id(p))
            yield p


def build_model_from_config_dict(cfg_dict):
    class DictConfig:
        pass

    cfg = DictConfig()
    data_cfg = cfg_dict.get("data", cfg_dict)
    model_cfg = cfg_dict.get("model", cfg_dict)

    image_size = data_cfg["image_size"]
    cfg.image_size = (image_size, image_size) if isinstance(image_size, int) else tuple(image_size)
    cfg.num_channels = data_cfg.get("num_channels", 1)
    cfg.model_type = model_cfg.get("model_type", "dctnn")
    cfg.encoders = model_cfg.get("encoders", ["patch", "patch", "patch"])
    cfg.reconformer_num_ch = tuple(model_cfg.get("reconformer_num_ch", (96, 48, 24)))
    cfg.reconformer_num_iter = model_cfg.get("reconformer_num_iter", 5)
    cfg.reconformer_down_scales = tuple(model_cfg.get("reconformer_down_scales", (2.0, 1.0, 1.5)))
    cfg.reconformer_num_heads = tuple(model_cfg.get("reconformer_num_heads", (6, 6, 6)))
    cfg.reconformer_depths = tuple(model_cfg.get("reconformer_depths", (2, 1, 1)))
    cfg.reconformer_window_sizes = tuple(model_cfg.get("reconformer_window_sizes", (8, 8, 8)))
    cfg.reconformer_mlp_ratio = model_cfg.get("reconformer_mlp_ratio", 2.0)
    cfg.reconformer_resi_connection = model_cfg.get("reconformer_resi_connection", "1conv")
    cfg.reconformer_use_checkpoint = tuple(model_cfg.get(
        "reconformer_use_checkpoint", (False, False, True, True, False, False)
    ))
    patch_size = model_cfg.get("patch_size", (16, 16))
    cfg.patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else tuple(patch_size)
    cfg.apt_layout = model_cfg.get("apt_layout")
    cfg.apt_embed_dim = model_cfg.get("apt_embed_dim", 256)
    apt_ref_grid = model_cfg.get("apt_rope_ref_grid")
    cfg.apt_rope_ref_grid = None if apt_ref_grid is None else tuple(apt_ref_grid)
    cfg.apt_use_abs_pos_emb = model_cfg.get("apt_use_abs_pos_emb", False)
    cfg.nhead_patch = model_cfg.get("nhead_patch", 8)
    cfg.nhead_axial = model_cfg.get("nhead_axial", 8)
    cfg.layer_no = model_cfg["layer_no"]
    cfg.num_encoder_layers = model_cfg["num_encoder_layers"]
    cfg.layer_norm_eps = model_cfg.get("layer_norm_eps", 1e-5)
    cfg.learning = model_cfg.get("learning", "k_space")
    cfg.lambda_schedule = model_cfg.get("lambda_schedule", "none")
    cfg.pos_emb_type = model_cfg.get("pos_emb_type", "APE")
    cfg.attn_type = model_cfg.get("attn_type", "standard")
    cfg.rope_theta = model_cfg.get("rope_theta", 100.0)
    cfg.rope_mixed_rotate = model_cfg.get("rope_mixed_rotate", True)
    cfg.axial_row_stride = model_cfg.get("axial_row_stride", 1)
    cfg.mask_vertical_attn = model_cfg.get("mask_vertical_attn", "none")
    cfg.ffn_sharing = model_cfg.get("ffn_sharing", "none")
    cfg.flattening_order = model_cfg.get("flattening_order", "row_major")
    return _build_model_impl(cfg)


_DEFAULT_CENTER_FRACTIONS = {
    4: 0.08,
    8: 0.04,
}


def _default_center_fraction(accel):
    return _DEFAULT_CENTER_FRACTIONS.get(int(accel), 0.04)


class FastMRIMaskGenerator:
    def __init__(self, accelerations, center_fractions=None, mask_type="random"):
        if center_fractions is not None and len(center_fractions) != len(accelerations):
            raise ValueError("center_fractions must be None or match acceleration_factors length")

        if mask_type == "random":
            mask_cls = RandomMaskFunc
        elif mask_type in {"equispaced", "equi_spaced"}:
            mask_cls = EquiSpacedMaskFunc
        else:
            raise ValueError("mask_type must be 'random' or 'equispaced'")

        if center_fractions is None:
            center_fractions = [_default_center_fraction(accel) for accel in accelerations]

        self.center_fractions = {
            int(accel): float(center_fraction)
            for accel, center_fraction in zip(accelerations, center_fractions)
        }
        self.mask_funcs = {
            int(accel): mask_cls(
                center_fractions=[self.center_fractions[int(accel)]],
                accelerations=[int(accel)],
            )
            for accel in accelerations
        }

    def apply(self, kspace_full, accel, seed=None):
        accel = int(accel)
        if accel not in self.mask_funcs:
            raise ValueError(f"Acceleration R={accel} was not configured")

        device = kspace_full.device
        # fastmri's apply_mask creates its mask on CPU, so run on CPU then restore device
        kspace_ri = torch.view_as_real(kspace_full).cpu()
        masked_ri, mask, num_low_frequencies = apply_mask(
            kspace_ri,
            self.mask_funcs[accel],
            seed=seed,
        )
        masked_kspace = torch.view_as_complex(masked_ri.contiguous()).to(device)
        mask = mask.squeeze(-1).to(device=device, dtype=kspace_full.real.dtype)
        return masked_kspace, mask, num_low_frequencies


_NORMALIZERS = {
    "zscore":              _norm.zscore,
    "kspace_companding":   _norm.kspace_companding,
    "log_kspace":          _norm.log_kspace,
    "fastmri_magnitude":   _norm.fastmri_magnitude,
    "reconformer":         _norm.reconformer,
    "robust_shifted":      _norm.robust_shifted,
    None:                  _norm.none,
    "none":                _norm.none,
}


KSPACE_FILL_STRATEGIES = {"linear", "cartesian_linear", "exponential", "inverse_distance"}


def _mask_to_2d(mask):
    """Collapse a broadcastable sampling mask to [H, W] for the interpolator."""
    if mask.ndim == 2:
        return mask
    squeezed = mask.squeeze()
    if squeezed.ndim == 2:
        return squeezed
    if squeezed.ndim == 1:
        h, w = squeezed.numel(), squeezed.numel()
        return squeezed.view(1, w).expand(h, w)
    raise ValueError(f"Cannot interpret mask with shape {tuple(mask.shape)} as a 2D Cartesian mask")


@torch.no_grad()
def interpolate_kspace_us(kspace_us, mask, strategy):
    """
    Replace the zero-filled unmeasured k-space with an interpolated estimate.

    Only Cartesian full-column masks are supported (see DcTNN/radial_interpolation.py).
    Measured points are preserved exactly; only unmeasured points are filled. The returned
    tensor is only meant as the MODEL INPUT — data consistency still uses the raw measured
    k-space (simulate_undersampling returns the measured k-space as the DC input).
    """
    if strategy not in KSPACE_FILL_STRATEGIES:
        raise ValueError(
            f"Unknown kspace_fill strategy '{strategy}'. Choose from: {sorted(KSPACE_FILL_STRATEGIES)}"
        )
    from DcTNN.radial_interpolation import radial_complex_interpolate

    if kspace_us.ndim != 4:
        raise ValueError(f"Expected [B, 1, H, W] k-space, got {tuple(kspace_us.shape)}")
    mask2d = _mask_to_2d(mask).to(device=kspace_us.device)
    filled = []
    for index in range(kspace_us.shape[0]):
        sample = kspace_us[index, 0]
        interp, _ = radial_complex_interpolate(sample, mask2d, strategy=strategy)
        filled.append(interp.unsqueeze(0))
    return torch.stack(filled, dim=0)


def simulate_undersampling(
    kspace_full,
    mask,
    learning="k_space",
    norm="none",
    kspace_us=None,
    robust_clip: float = 3.0,
    robust_shift: float = 3.0,
    companding_p: float = 0.8,
    companding_a: float = 0.5,
    companding_centering: str = "fft",
    kspace_fill: str | None = None,
):
    """
    learning="complex_image" : preserve complex image values through the model and FFT data consistency
    norm="zscore" : z-score normalise real/imag separately using undersampled image stats
    norm="robust_shifted" : median/IQR scale, smooth clip, and shift in the learning domain
    norm="kspace_companding" : radial magnitude companding in k-space (k_space learning only)
    norm="log_kspace" : log1p magnitude k-space normalization with preserved phase (k_space learning only)
    norm=None     : no normalisation — tensors left in raw k-space units
    kspace_fill  : pre-fill the unmeasured k-space before normalisation instead of zero-filling.
                   One of "linear", "cartesian_linear", "exponential", "inverse_distance"
                   (see DcTNN/radial_interpolation.py). None/"none"/"zero_fill" = zero-fill.
                   The filled k-space is used as the model input; the raw measured k-space is
                   still returned as the DC input.
    """
    measured = kspace_full * mask if kspace_us is None else kspace_us
    prefilled = kspace_fill not in (None, "none", "zero_fill")
    if prefilled:
        model_kspace = interpolate_kspace_us(measured, mask, kspace_fill)
    else:
        model_kspace = measured
    fn = _NORMALIZERS.get(norm)
    if fn is None:
        raise ValueError(f"Unknown norm '{norm}'. Choose from: {list(_NORMALIZERS)}")
    model_input, _, target, metric = fn(
        kspace_full,
        mask,
        learning,
        kspace_us=model_kspace,
        kspace_prefilled=prefilled,
        robust_clip=robust_clip,
        robust_shift=robust_shift,
        companding_p=companding_p,
        companding_a=companding_a,
        companding_centering=companding_centering,
    )
    # Data consistency always blends with the raw measured k-space (the interpolated
    # points are only a model-input initialisation and must not be re-imposed by DC).
    return model_input, measured, target, metric
