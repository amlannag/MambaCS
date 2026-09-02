"""
Shared helpers used by training and inference.
"""

import torch
from fastmri.data.subsample import EquiSpacedMaskFunc, RandomMaskFunc
from fastmri.data.transforms import apply_mask

from DcTNN.model import TokenVIT, axVIT, CrossAttentionVIT, cascadeNet
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
    "axial": lambda cfg: (
        axVIT,
        dict(
            layerNo=cfg.layer_no,
            numCh=cfg.num_channels,
            d_model=None,
            nhead=cfg.nhead_axial,
            num_encoder_layers=cfg.num_encoder_layers,
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
            dim_feedforward=None,
            d_model=None,
            pos_emb_type=cfg.pos_emb_type,
            rope_theta=cfg.rope_theta,
            rope_mixed_rotate=cfg.rope_mixed_rotate,
            attn_type=cfg.attn_type,
        ),
    ),
}


def _build_model_impl(cfg):
    if cfg.model_type == "reconformer":
        if cfg.learning != "complex_image":
            raise ValueError("ReconFormer requires learning='complex_image'")
        image_size = tuple(int(value) for value in cfg.image_size)
        if len(image_size) != 2 or image_size[0] != image_size[1]:
            raise ValueError(f"ReconFormer requires a square image_size, got {image_size}")
        return ReconFormerBaseline(
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

    enc_list = []
    enc_args = []
    for name in cfg.encoders:
        if name not in _ENCODER_ARGS:
            raise ValueError(f"Unknown encoder '{name}'. Choose from: {list(_ENCODER_ARGS)}")
        cls, args = _ENCODER_ARGS[name](cfg)
        args["numCh"] = num_ch
        enc_list.append(cls)
        enc_args.append(args)

    use_learned_lamb = cfg.lambda_schedule == "none"
    return cascadeNet(
        cfg.image_size,
        enc_list,
        enc_args,
        use_learned_lamb,
        learning=cfg.learning,
    )


def build_model(cfg):
    return _build_model_impl(cfg)


def build_model_from_config_dict(cfg_dict):
    class DictConfig:
        pass

    cfg = DictConfig()
    data_cfg = cfg_dict["data"]
    model_cfg = cfg_dict["model"]

    cfg.image_size = data_cfg["image_size"]
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
    cfg.patch_size = model_cfg["patch_size"]
    cfg.nhead_patch = model_cfg["nhead_patch"]
    cfg.nhead_axial = model_cfg["nhead_axial"]
    cfg.layer_no = model_cfg["layer_no"]
    cfg.num_encoder_layers = model_cfg["num_encoder_layers"]
    cfg.learning = model_cfg.get("learning", "k_space")
    cfg.lambda_schedule = model_cfg.get("lambda_schedule", "none")
    cfg.pos_emb_type = model_cfg.get("pos_emb_type", "APE")
    cfg.attn_type = model_cfg.get("attn_type", "standard")
    cfg.rope_theta = model_cfg.get("rope_theta", 100.0)
    cfg.rope_mixed_rotate = model_cfg.get("rope_mixed_rotate", True)
    cfg.axial_row_stride = model_cfg.get("axial_row_stride", 1)
    cfg.mask_vertical_attn = model_cfg.get("mask_vertical_attn", "none")
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
):
    """
    learning="complex_image" : preserve complex image values through the model and FFT data consistency
    norm="zscore" : z-score normalise real/imag separately using undersampled image stats
    norm="robust_shifted" : median/IQR scale, smooth clip, and shift in the learning domain
    norm="kspace_companding" : radial magnitude companding in k-space (k_space learning only)
    norm="log_kspace" : log1p magnitude k-space normalization with preserved phase (k_space learning only)
    norm=None     : no normalisation — tensors left in raw k-space units
    """
    fn = _NORMALIZERS.get(norm)
    if fn is None:
        raise ValueError(f"Unknown norm '{norm}'. Choose from: {list(_NORMALIZERS)}")
    return fn(
        kspace_full,
        mask,
        learning,
        kspace_us=kspace_us,
        robust_clip=robust_clip,
        robust_shift=robust_shift,
        companding_p=companding_p,
        companding_a=companding_a,
        companding_centering=companding_centering,
    )
