"""
Shared helpers used by training and inference.
"""

import numpy as np
import torch
from fastmri.data.subsample import EquiSpacedMaskFunc, RandomMaskFunc
from fastmri.data.transforms import apply_mask

from DcTNN.model import TokenVIT, axVIT, CrossAttentionVIT, CrossAxialNeighbourhoodVIT, FNetVIT, FixedAPTVIT, PCAVIT, cascadeNet
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
    "fastmri_brain": (
        "/scratch/user/uqanag/fastmri_brain/train",
        "/scratch/user/uqanag/fastmri_brain/val",
    ),
}

# Datasets stored as fastMRI-style HDF5 volumes (one .h5 per volume, `kspace` dataset); everything else is PNG slices.
H5_DATASETS = ("fastmri", "fastmri_brain")
MULTICOIL_DATASETS = ("fastmri_brain",)


def is_h5_dataset(cfg) -> bool:
    return cfg.dataset in H5_DATASETS


def is_multicoil_dataset(cfg) -> bool:
    return cfg.dataset in MULTICOIL_DATASETS


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


def _kspace_stem_args(cfg):
    """Options for the optional k-space stems (GlobalFilter / SpiritConvStem / KSpaceConvStem) on patch and axial encoders."""
    kernels = getattr(cfg, "spirit_conv_kernels", (5, 3))
    if isinstance(kernels, int):
        kernels = (kernels,)
    else:
        kernels = tuple(int(k) for k in kernels)
    return dict(
        global_filter=getattr(cfg, "global_filter", False),
        global_filter_mid=getattr(cfg, "global_filter_mid", False),
        kspace_conv=getattr(cfg, "kspace_conv", False),
        kspace_conv_channels=getattr(cfg, "kspace_conv_channels", 8),
        kspace_conv_kernel=getattr(cfg, "kspace_conv_kernel", 3),
        spirit_conv=getattr(cfg, "spirit_conv", False),
        spirit_conv_kernels=kernels,
        spirit_conv_mode=getattr(cfg, "spirit_conv_mode", "gated"),
        image_conv=getattr(cfg, "image_conv", False),
        image_conv_channels=getattr(cfg, "image_conv_channels", 32),
        image_conv_layers=getattr(cfg, "image_conv_layers", 3),
        image_conv_kernel=getattr(cfg, "image_conv_kernel", 3),
    )


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
            d_model=getattr(cfg, "axial_d_model", None),
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
            token_axis=getattr(cfg, "axial_token_axis", "both"),
            attn_scales=tuple(getattr(cfg, "attn_scales", (1, 3))),
            kspace_stem=_kspace_stem_args(cfg),
        ),
    ),
    "cross_axial": lambda cfg: (
        CrossAttentionVIT,
        dict(
            layerNo=cfg.layer_no,
            numCh=cfg.num_channels,
            d_model=getattr(cfg, "axial_d_model", None),
            nhead=cfg.nhead_axial,
            num_encoder_layers=cfg.num_encoder_layers,
            layer_norm_eps=getattr(cfg, "layer_norm_eps", 1e-5),
            dim_feedforward=None,
            pos_emb_type=cfg.pos_emb_type,
            rope_theta=cfg.rope_theta,
            rope_mixed_rotate=cfg.rope_mixed_rotate,
            # Cross stages only implement plain complex attention; multi-scale Q/K applies to the axial stages.
            attn_type="complex" if cfg.attn_type == "complex_ms" else cfg.attn_type,
            row_stride=cfg.axial_row_stride,
            kspace_stem=_kspace_stem_args(cfg),
            coil_mode=getattr(cfg, "cross_axial_coil_mode", "folded"),
        ),
    ),
    "cross_axial_neighbourhood": lambda cfg: (
        CrossAxialNeighbourhoodVIT,
        dict(
            layerNo=cfg.layer_no,
            numCh=cfg.num_channels,
            d_model=getattr(cfg, "axial_d_model", None),
            nhead=cfg.nhead_axial,
            num_encoder_layers=cfg.num_encoder_layers,
            layer_norm_eps=getattr(cfg, "layer_norm_eps", 1e-5),
            dim_feedforward=None,
            pos_emb_type=cfg.pos_emb_type,
            rope_theta=cfg.rope_theta,
            rope_mixed_rotate=cfg.rope_mixed_rotate,
            # The global branch is plain complex cross-attention (as "cross_axial"); the local branch's scoring
            # rule is neighbourhood_attn_type.
            attn_type="complex" if cfg.attn_type == "complex_ms" else cfg.attn_type,
            row_stride=cfg.axial_row_stride,
            kspace_stem=_kspace_stem_args(cfg),
            strip_len=getattr(cfg, "neighbourhood_strip_len", 16),
            n_neighbours=getattr(cfg, "neighbourhood_n_neighbours", 3),
            kx_halo=getattr(cfg, "neighbourhood_kx_halo", 1),
            keys_acquired_only=getattr(cfg, "neighbourhood_keys_acquired_only", True),
            local_attn_type=getattr(cfg, "neighbourhood_attn_type", "complex"),
            combine=getattr(cfg, "neighbourhood_combine", "sum"),
            local_layers=getattr(cfg, "neighbourhood_layers", None),
            global_coil_mode=getattr(cfg, "cross_axial_coil_mode", "folded"),
        ),
    ),
    "fnet": lambda cfg: (
        FNetVIT,
        dict(
            layerNo=cfg.layer_no,
            numCh=cfg.num_channels,
            d_model=getattr(cfg, "axial_d_model", None),
            nhead=cfg.nhead_axial,
            num_encoder_layers=cfg.num_encoder_layers,
            layer_norm_eps=getattr(cfg, "layer_norm_eps", 1e-5),
            dim_feedforward=None,
            pos_emb_type=cfg.pos_emb_type,
            rope_theta=cfg.rope_theta,
            rope_mixed_rotate=cfg.rope_mixed_rotate,
            attn_type=cfg.attn_type,
            row_stride=cfg.axial_row_stride,
            fft_norm=getattr(cfg, "fnet_fft_norm", "ortho"),
            token_axis=getattr(cfg, "fnet_token_axis", "vertical"),
            with_embedding=getattr(cfg, "fnet_with_embedding", True),
            share_ffn=getattr(cfg, "fnet_share_ffn", False),
            kspace_stem=_kspace_stem_args(cfg),
        ),
    ),
    "pca": lambda cfg: (
        PCAVIT,
        dict(
            layerNo=cfg.layer_no,
            numCh=cfg.num_channels,
            d_model=None,
            nhead=getattr(cfg, "pca_nhead", 8),
            num_encoder_layers=cfg.num_encoder_layers,
            layer_norm_eps=getattr(cfg, "layer_norm_eps", 1e-5),
            dim_feedforward=None,
            pos_emb_type=cfg.pos_emb_type,
            rope_theta=cfg.rope_theta,
            rope_mixed_rotate=cfg.rope_mixed_rotate,
            attn_type=cfg.attn_type,
            row_stride=cfg.axial_row_stride,
            tokenizer=getattr(cfg, "pca_tokenizer", "axial"),
            n_bins=getattr(cfg, "pca_bins", 3),
            bin_rule=getattr(cfg, "pca_bin_rule", "equal_variance"),
            detach_basis=getattr(cfg, "pca_detach_basis", True),
            center=getattr(cfg, "pca_center", True),
            layers_per_bin=getattr(cfg, "pca_layers_per_bin", 1),
            layers_after_merge=getattr(cfg, "pca_layers_after_merge", 1),
            scope=getattr(cfg, "pca_scope", "volume"),
            apt_layout=getattr(cfg, "apt_layout", None),
            apt_embed_dim=getattr(cfg, "apt_embed_dim", 256),
            apt_rope_ref_grid=getattr(cfg, "apt_rope_ref_grid", None),
            apt_use_abs_pos_emb=getattr(cfg, "apt_use_abs_pos_emb", False),
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
            attn_scales=tuple(getattr(cfg, "attn_scales", (1, 3))),
            kspace_stem=_kspace_stem_args(cfg),
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
    if getattr(cfg, "image_conv", False) and cfg.learning != "k_space":
        raise ValueError("image_conv (image-domain CNN stem) requires learning='k_space'")
    real_channels = getattr(cfg, "kspace_real_channels", False)
    if real_channels:
        if cfg.learning != "k_space":
            raise ValueError("kspace_real_channels requires learning='k_space'")
        if cfg.attn_type in _COMPLEX_ATTN_TYPES:
            raise ValueError("kspace_real_channels needs a real-valued attn_type ('standard'); "
                             f"got {cfg.attn_type!r}")
        unsupported = {"fixed_apt", "cross_axial", "cross_axial_neighbourhood", "pca"} & set(cfg.encoders)
        if unsupported:
            raise ValueError(f"kspace_real_channels does not support complex-only encoders: {sorted(unsupported)}")
        num_ch = 2 * num_ch

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
        real_channels=real_channels,
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
    cfg.kspace_real_channels = model_cfg.get("kspace_real_channels", False)
    cfg.lambda_schedule = model_cfg.get("lambda_schedule", "none")
    cfg.pos_emb_type = model_cfg.get("pos_emb_type", "APE")
    cfg.attn_type = model_cfg.get("attn_type", "standard")
    cfg.attn_scales = tuple(model_cfg.get("attn_scales", (1, 3)))
    cfg.fnet_share_ffn = model_cfg.get("fnet_share_ffn", False)
    cfg.global_filter = model_cfg.get("global_filter", False)
    cfg.global_filter_mid = model_cfg.get("global_filter_mid", False)
    cfg.kspace_conv = model_cfg.get("kspace_conv", False)
    cfg.kspace_conv_channels = model_cfg.get("kspace_conv_channels", 8)
    cfg.kspace_conv_kernel = model_cfg.get("kspace_conv_kernel", 3)
    cfg.image_conv = model_cfg.get("image_conv", False)
    cfg.image_conv_channels = model_cfg.get("image_conv_channels", 32)
    cfg.image_conv_layers = model_cfg.get("image_conv_layers", 3)
    cfg.image_conv_kernel = model_cfg.get("image_conv_kernel", 3)
    cfg.rope_theta = model_cfg.get("rope_theta", 100.0)
    cfg.rope_mixed_rotate = model_cfg.get("rope_mixed_rotate", True)
    cfg.axial_row_stride = model_cfg.get("axial_row_stride", 1)
    cfg.axial_token_axis = model_cfg.get("axial_token_axis", "both")
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
    "log_quantile":        _norm.log_quantile,
    "fastmri_magnitude":   _norm.fastmri_magnitude,
    "image_magnitude":     _norm.image_magnitude,
    "reconformer":         _norm.reconformer,
    "robust_shifted":      _norm.robust_shifted,
    None:                  _norm.none,
    "none":                _norm.none,
}


KSPACE_FILL_STRATEGIES = {"linear", "cartesian_linear", "exponential", "inverse_distance", "grappa"}


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


def _grappa_source_offsets(target_column, acquired_columns, source_line_count, acs_width):
    candidates = []
    for start in range(acquired_columns.size - source_line_count + 1):
        sources = acquired_columns[start:start + source_line_count]
        offsets = tuple((sources - target_column).astype(int).tolist())
        calibration_start = max(0, -min(offsets))
        calibration_stop = min(acs_width - 1, acs_width - 1 - max(offsets))
        if calibration_start <= calibration_stop:
            candidates.append((int(abs(sources - target_column).sum()), offsets))
    return min(candidates)[1] if candidates else None


def _grappa_source_features(kspace, target_column, phase_offsets, readout_offsets):
    _, height, width = kspace.shape
    rows = np.arange(height)
    features = []
    for phase_offset in phase_offsets:
        column = target_column + phase_offset
        if not 0 <= column < width:
            raise ValueError(f"GRAPPA source column {column} is outside k-space.")
        for readout_offset in readout_offsets:
            source_rows = np.clip(rows + readout_offset, 0, height - 1)
            features.append(kspace[:, source_rows, column].T)
    return np.concatenate(features, axis=1).astype(np.complex128, copy=False)


def _grappa_calibrate_pattern(kspace, acs_start, acs_stop, phase_offsets, readout_offsets, ridge):
    _, height, _ = kspace.shape
    targets = [
        column for column in range(acs_start, acs_stop)
        if all(acs_start <= column + offset < acs_stop for offset in phase_offsets)
    ]
    if not targets:
        raise ValueError(f"No ACS training targets for GRAPPA source offsets {phase_offsets}.")

    x = np.concatenate([
        _grappa_source_features(kspace, column, phase_offsets, readout_offsets)
        for column in targets
    ], axis=0)
    y = np.concatenate([kspace[:, :, column].T for column in targets], axis=0)
    y = y.astype(np.complex128, copy=False)
    gram = x.conj().T @ x
    scale = max(float(gram.diagonal().real.mean()), np.finfo(np.float64).eps)
    regularized = gram + ridge * scale * np.eye(gram.shape[0], dtype=np.complex128)
    return np.linalg.solve(regularized, x.conj().T @ y)


def _grappa_prefill_sample(kspace_us, line_mask, num_low_frequencies,
                           phase_line_count, readout_kernel_width, ridge):
    coils, height, width = kspace_us.shape
    if phase_line_count < 1:
        raise ValueError("grappa_phase_lines must be positive.")
    if readout_kernel_width < 1 or readout_kernel_width % 2 != 1:
        raise ValueError("grappa_readout_width must be an odd positive number.")
    if ridge < 0:
        raise ValueError("grappa_ridge must be non-negative.")

    acs_start = (width - num_low_frequencies + 1) // 2
    acs_stop = acs_start + num_low_frequencies
    if not 0 < num_low_frequencies <= width or not np.all(line_mask[acs_start:acs_stop]):
        raise ValueError("The GRAPPA ACS region must be fully sampled and fit the phase-encode width.")

    acquired_columns = np.flatnonzero(line_mask)
    missing_columns = np.flatnonzero(~line_mask)
    patterns = {
        int(column): offsets
        for column in missing_columns
        if (offsets := _grappa_source_offsets(
            int(column), acquired_columns, phase_line_count, acs_stop - acs_start,
        )) is not None
    }
    readout_radius = readout_kernel_width // 2
    readout_offsets = tuple(range(-readout_radius, readout_radius + 1))
    weights = {
        offsets: _grappa_calibrate_pattern(
            kspace_us, acs_start, acs_stop, offsets, readout_offsets, ridge,
        )
        for offsets in set(patterns.values())
    }

    reconstructed = kspace_us.copy()
    for column, offsets in patterns.items():
        features = _grappa_source_features(kspace_us, column, offsets, readout_offsets)
        reconstructed[:, :, column] = (features @ weights[offsets]).T
    return reconstructed


def _grappa_batch_features(kspace, target_column, phase_offsets, readout_offsets):
    _, _, height, width = kspace.shape
    rows = np.arange(height)
    features = []
    for phase_offset in phase_offsets:
        column = target_column + phase_offset
        if not 0 <= column < width:
            raise ValueError(f"GRAPPA source column {column} is outside k-space.")
        for readout_offset in readout_offsets:
            source_rows = np.clip(rows + readout_offset, 0, height - 1)
            features.append(kspace[:, :, source_rows, column].transpose(0, 2, 1))
    return np.concatenate(features, axis=2).astype(np.complex128, copy=False)


def _grappa_prefill_batch(kspace_us, line_mask, num_low_frequencies,
                          phase_line_count, readout_kernel_width, ridge):
    _, _, _, width = kspace_us.shape
    acs_start = (width - num_low_frequencies + 1) // 2
    acs_stop = acs_start + num_low_frequencies
    if not 0 < num_low_frequencies <= width or not np.all(line_mask[acs_start:acs_stop]):
        raise ValueError("The GRAPPA ACS region must be fully sampled and fit the phase-encode width.")

    acquired_columns = np.flatnonzero(line_mask)
    missing_columns = np.flatnonzero(~line_mask)
    patterns = {
        int(column): offsets
        for column in missing_columns
        if (offsets := _grappa_source_offsets(
            int(column), acquired_columns, phase_line_count, acs_stop - acs_start,
        )) is not None
    }
    readout_radius = readout_kernel_width // 2
    readout_offsets = tuple(range(-readout_radius, readout_radius + 1))
    weights = {}
    identity = None
    for offsets in set(patterns.values()):
        targets = [
            column for column in range(acs_start, acs_stop)
            if all(acs_start <= column + offset < acs_stop for offset in offsets)
        ]
        if not targets:
            raise ValueError(f"No ACS training targets for GRAPPA source offsets {offsets}.")
        x = np.concatenate([
            _grappa_batch_features(kspace_us, column, offsets, readout_offsets)
            for column in targets
        ], axis=1)
        y = np.concatenate([
            kspace_us[:, :, :, column].transpose(0, 2, 1)
            for column in targets
        ], axis=1).astype(np.complex128, copy=False)
        gram = x.conj().transpose(0, 2, 1) @ x
        scale = np.maximum(
            gram.diagonal(axis1=1, axis2=2).real.mean(axis=1),
            np.finfo(np.float64).eps,
        )
        if identity is None or identity.shape[0] != gram.shape[1]:
            identity = np.eye(gram.shape[1], dtype=np.complex128)
        regularized = gram + ridge * scale[:, None, None] * identity
        rhs = x.conj().transpose(0, 2, 1) @ y
        weights[offsets] = np.linalg.solve(regularized, rhs)

    reconstructed = kspace_us.copy()
    for column, offsets in patterns.items():
        features = _grappa_batch_features(kspace_us, column, offsets, readout_offsets)
        reconstructed[:, :, :, column] = (features @ weights[offsets]).transpose(0, 2, 1)
    return reconstructed


@torch.no_grad()
def grappa_prefill_kspace_us(kspace_us, mask, num_low_frequencies,
                             phase_line_count=2, readout_kernel_width=3, ridge=1e-1):
    if kspace_us.ndim != 4 or kspace_us.shape[1] < 2:
        raise ValueError(f"GRAPPA prefill expects multicoil [B, C>=2, H, W] k-space, got {tuple(kspace_us.shape)}")
    batch_size, _, _, width = kspace_us.shape
    line_masks = mask.detach().to(device="cpu", dtype=torch.bool).reshape(-1, width).numpy()
    if line_masks.shape[0] == 1:
        line_masks = np.repeat(line_masks, batch_size, axis=0)
    elif line_masks.shape[0] != batch_size:
        raise ValueError(f"Mask batch dimension {line_masks.shape[0]} does not match k-space batch {batch_size}.")

    low_frequencies = torch.as_tensor(num_low_frequencies).detach().cpu().reshape(-1).numpy()
    if low_frequencies.size == 1:
        low_frequencies = np.repeat(low_frequencies, batch_size)
    elif low_frequencies.size != batch_size:
        raise ValueError("num_low_frequencies must be scalar or contain one value per batch item.")

    samples = kspace_us.detach().cpu().numpy()
    if np.all(line_masks == line_masks[0]) and np.all(low_frequencies == low_frequencies[0]):
        filled = _grappa_prefill_batch(
            samples,
            line_masks[0],
            int(low_frequencies[0]),
            phase_line_count,
            readout_kernel_width,
            ridge,
        )
    else:
        filled = np.stack([
            _grappa_prefill_sample(
                samples[index],
                line_masks[index],
                int(low_frequencies[index]),
                phase_line_count,
                readout_kernel_width,
                ridge,
            )
            for index in range(batch_size)
        ])
    return torch.as_tensor(filled, device=kspace_us.device, dtype=kspace_us.dtype)


@torch.no_grad()
def interpolate_kspace_us(kspace_us, mask, strategy, num_low_frequencies=None,
                          grappa_phase_lines=2, grappa_readout_width=3, grappa_ridge=1e-1):
    """
    Replace the zero-filled unmeasured k-space with an interpolated estimate.

    Measured points are preserved exactly; only unmeasured points are filled. The returned
    tensor is only meant as the MODEL INPUT — data consistency still uses the raw measured
    k-space (simulate_undersampling returns the measured k-space as the DC input).
    """
    if strategy not in KSPACE_FILL_STRATEGIES:
        raise ValueError(
            f"Unknown kspace_fill strategy '{strategy}'. Choose from: {sorted(KSPACE_FILL_STRATEGIES)}"
        )
    if strategy == "grappa":
        if num_low_frequencies is None:
            raise ValueError("num_low_frequencies is required for GRAPPA prefill.")
        return grappa_prefill_kspace_us(
            kspace_us, mask, num_low_frequencies, grappa_phase_lines, grappa_readout_width, grappa_ridge,
        )

    from DcTNN.radial_interpolation import radial_complex_interpolate

    if kspace_us.ndim != 4 or kspace_us.shape[1] != 1:
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
    volume_scale=None,
    sigma_raw=None,
    norm_quantile: float = 0.95,
    num_low_frequencies=None,
    grappa_phase_lines: int = 2,
    grappa_readout_width: int = 3,
    grappa_ridge: float = 1e-1,
):
    """
    volume_scale : optional [B] volume-wise normalisation scale (fastmri_magnitude / log_quantile, k_space learning):
                   replaces the per-slice quantile. norm_quantile : quantile of the zero-filled magnitude pinned to 1
                   by fastmri_magnitude / log_quantile (0.95 default, 1.0 = max). sigma_raw : optional [B] raw k-space noise std per sample, forwarded into stats
                   ("sigma_raw" / "sigma_norm") for SNR-weighted losses.
    learning="complex_image" : preserve complex image values through the model and FFT data consistency
    norm="zscore" : z-score normalise real/imag separately using undersampled image stats
    norm="image_magnitude" : divide the complex zero-filled and GT images by the norm_quantile of |IFFT(k_zf)|, then
                   map back to the learning domain (k_space input = k / q_img); GT shares the same scale
    norm="robust_shifted" : median/IQR scale, smooth clip, and shift in the learning domain
    norm="kspace_companding" : radial magnitude companding in k-space (k_space learning only)
    norm="log_kspace" : log1p magnitude k-space normalization with preserved phase (k_space learning only)
    norm="log_quantile" : log1p(|k| / q_p) with preserved phase, q_p = norm_quantile of the zero-filled |k| (k_space only)
    norm=None     : no normalisation — tensors left in raw k-space units
    kspace_fill  : pre-fill the unmeasured k-space before normalisation instead of zero-filling.
                   One of "linear", "cartesian_linear", "exponential", "inverse_distance", "grappa",
                   or None/"none"/"zero_fill" = zero-fill. The filled k-space is used as the model
                   input; the raw measured k-space is still returned as the DC input.
    """
    measured = kspace_full * mask if kspace_us is None else kspace_us
    prefilled = kspace_fill not in (None, "none", "zero_fill")
    if prefilled:
        model_kspace = interpolate_kspace_us(
            measured,
            mask,
            kspace_fill,
            num_low_frequencies=num_low_frequencies,
            grappa_phase_lines=grappa_phase_lines,
            grappa_readout_width=grappa_readout_width,
            grappa_ridge=grappa_ridge,
        )
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
        volume_scale=volume_scale,
        sigma_raw=sigma_raw,
        quantile=norm_quantile,
    )
    # Data consistency always blends with the raw measured k-space (the interpolated
    # points are only a model-input initialisation and must not be re-imposed by DC).
    return model_input, measured, target, metric
