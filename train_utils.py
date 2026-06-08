"""
Shared helpers used by training and inference.
"""

import torch

from DcTNN.dc import fft_2d, ifft_2d
from DcTNN.model import TokenVIT, axVIT, cascadeNet


def resolve_data_dirs(cfg):
    if cfg.val_data_dir:
        return cfg.data_dir, cfg.val_data_dir
    return cfg.data_dir, cfg.data_dir


def psnr(pred, target, max_val=1.0):
    mse = torch.mean(torch.abs(pred - target) ** 2)
    if mse == 0:
        return torch.tensor(float("inf"))
    return 20.0 * torch.log10(torch.tensor(max_val, device=pred.device) / torch.sqrt(mse))


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
    num_ch = cfg.num_channels

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
    cfg.encoders = model_cfg.get("encoders", ["patch", "patch", "patch"])
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
    return _build_model_impl(cfg)


def generate_column_mask(image_size, accel, device):
    H, W = image_size if isinstance(image_size, (tuple, list)) else (image_size, image_size)

    n_total = W // accel
    beta  = 0.08 if accel <= 4 else 0.04   # 8% ACS for R<=4, 4% for R>4
    n_acs = min(round(beta * W), n_total)
    n_outer = n_total - n_acs

    center = W // 2
    acs_start = center - n_acs // 2
    acs_cols = torch.arange(acs_start, acs_start + n_acs, device=device)

    mask = torch.zeros(H, W, device=device)
    mask[:, acs_cols] = 1.0

    if n_outer > 0:
        # Build outer column indices (everything outside the ACS band)
        all_cols = torch.arange(W, dtype=torch.float32, device=device)
        outer_mask = torch.ones(W, dtype=torch.bool, device=device)
        outer_mask[acs_start : acs_start + n_acs] = False
        outer_cols = all_cols[outer_mask]

        # Gaussian weights — higher probability near k-space centre
        sigma = W / 4.0
        weights = torch.exp(-0.5 * ((outer_cols - center) / sigma) ** 2)
        weights = weights / weights.sum()

        sampled_idx = torch.multinomial(weights, min(n_outer, len(outer_cols)), replacement=False)
        mask[:, outer_cols[sampled_idx].long()] = 1.0

    return mask


def simulate_undersampling(kspace_full, mask, learning="k_space"):
    kspace_us  = kspace_full * mask
    img_us     = ifft_2d(kspace_us)               # complex zero-filled image
    img_gt_mag = torch.abs(ifft_2d(kspace_full))  # GT magnitude image (always needed)

    if learning == "k_space":
        real  = img_us.real
        mn    = real.amin(dim=(-2, -1), keepdim=True)
        scale = (real.amax(dim=(-2, -1), keepdim=True) - mn).clamp(min=1e-8)

        img_norm    = torch.complex((img_us.real - mn) / scale, img_us.imag)
        kspace_norm = fft_2d(img_norm)
        model_input = kspace_norm

    else:  # "image"
        img_us_mag = torch.abs(img_us)
        mn         = img_us_mag.amin(dim=(-2, -1), keepdim=True)
        scale      = (img_us_mag.amax(dim=(-2, -1), keepdim=True) - mn).clamp(min=1e-8)

        model_input = (img_us_mag - mn) / scale
        kspace_norm = fft_2d(model_input)          # DC reference: FFT of real magnitude input

    gt_norm = (img_gt_mag - mn) / scale            # always real magnitude in [0, 1]

    return model_input, kspace_norm, gt_norm, {"min": mn, "scale": scale}
