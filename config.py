"""
Config dataclass with default values for all hyperparameters.

To run experiments, define overrides in train_config.py and run:
    python train.py --exp_idx <N>

Encoder options for `encoders`:
    "cross_axial"   — vertical-only sampled/unsampled complex cross-attention
    "fnet"          — FNet Fourier token mixing over vertical column tokens (no attention)
    "axial"         — axial row/column transformer (global structure)
    "kaleidoscope"  — kaleidoscope patch transformer (non-local features)
    "patch"         — standard patch transformer (local texture)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class Config:

    # ---------------------------------------------------------------------------
    # Experiment identity
    # ---------------------------------------------------------------------------
    prefix: str = "MambaCS"
    name: str = "8x_acceleration"
    output_dir: str = "../Experiments"
    hpc_backend: str = "nvidia"

    # Set to a checkpoint path (e.g. "../Experiments/dctnn_baseline/latest.pth")
    # to resume a stopped run; leave as None to start fresh
    resume: Optional[str] = None

    # ---------------------------------------------------------------------------
    # Data
    # ---------------------------------------------------------------------------

    # "fastmri" — .h5 k-space files (centered IFFT, image-domain crop, centered FFT)
    # "oasis"   — PNG brain slices (converted to centered k-space via FFT)
    dataset: str = "fastmri"

    # Data directories — if None, auto-selected from DATASET_DIRS in train_utils.py
    data_dir: Optional[str] = None
    val_data_dir: Optional[str] = None

    kspace_key: str = "kspace"
    image_size: Tuple[int, int] = (320, 320)

    num_channels: int = 1

    acceleration_factors: List[int] = field(default_factory=lambda: [8])
    center_fractions: Optional[List[float]] = None
    mask_type: str = "random"

    val_fraction: float = 0.1
    seed: int = 42
    max_train_files: Optional[int] = None
    max_val_files: Optional[int] = 15

    # ---------------------------------------------------------------------------
    # Model architecture
    # ---------------------------------------------------------------------------

    # Ordered list of encoder stages in the cascade.
    # Options: "axial", "kaleidoscope", "patch"
    # Examples:
    #   ["axial", "kaleidoscope", "patch"]        — original DcTNN (3 stages)
    #   ["axial", "patch"]                        — 2-stage, no kaleidoscope
    #   ["patch", "patch", "patch"]               — patch-only ablation
    #   ["axial", "kaleidoscope", "patch", "patch"] — 4-stage deeper model
    model_type: str = "dctnn"
    encoders: List[str] = field(default_factory=lambda: ["patch", "patch", "patch"])
    reconformer_num_ch: Tuple[int, int, int] = (96, 48, 24)
    reconformer_num_iter: int = 5
    reconformer_down_scales: Tuple[float, float, float] = (2.0, 1.0, 1.5)
    reconformer_num_heads: Tuple[int, int, int] = (6, 6, 6)
    reconformer_depths: Tuple[int, int, int] = (2, 1, 1)
    reconformer_window_sizes: Tuple[int, int, int] = (8, 8, 8)
    reconformer_mlp_ratio: float = 2.0
    reconformer_resi_connection: str = "1conv"
    reconformer_use_checkpoint: Tuple[bool, bool, bool, bool, bool, bool] = (
        False, False, True, True, False, False
    )

    patch_size: tuple = (16, 16)
    apt_layout: Optional[dict] = None
    apt_embed_dim: int = 256
    apt_rope_ref_grid: Optional[Tuple[int, int]] = None
    apt_use_abs_pos_emb: bool = False
    axial_row_stride: int = 1
    nhead_patch: int = 8
    nhead_axial: int = 8
    layer_no: int = 1
    num_encoder_layers: int = 2
    # FNet stage ("fnet" in `encoders`): FFT normalisation of the Fourier token mixing.
    # "ortho" keeps |mix(x)| ~ |x| so the residual stream is preserved (recommended);
    # "backward" is the unnormalised FFT of the original FNet paper.
    fnet_fft_norm: str = "ortho"
    # Which axial tokens the FNet stage mixes over:
    #   "vertical"   — k-space columns are tokens (default; matches cross_axial / axial vertical branch)
    #   "horizontal" — k-space rows are tokens (row groups of axial_row_stride)
    #   "both"       — horizontal branch then vertical branch, like the axial encoder
    fnet_token_axis: str = "vertical"
    # Optional per-stage override of num_encoder_layers (one int per entry of `encoders`),
    # e.g. [1, 2, 1, 2] for a cross_axial/axial cascade with 1 layer per cross stage and 2 per axial.
    stage_encoder_layers: Optional[List[int]] = None
    layer_norm_eps: float = 1e-5
    learned_lambda: bool = True
    # Domain the model operates in: "k_space", "image", or "complex_image"
    learning: str = "k_space"
    # Normalisation: "zscore", "fastmri_magnitude", "robust_shifted", "kspace_companding", "log_kspace", or None
    norm: str = "zscore"
    # Pre-fill the undersampled (unmeasured) k-space BEFORE normalisation instead of
    # zero-filling: "linear", "cartesian_linear", "exponential", "inverse_distance"
    # (see DcTNN/radial_interpolation.py), or None/"zero_fill" for standard zero-fill.
    # Only Cartesian full-column masks are supported. The fill only affects the model input;
    # data consistency still uses the raw measured k-space.
    kspace_fill: Optional[str] = None
    robust_clip: float = 3.0
    robust_shift: float = 3.0
    companding_p: float = 0.8
    companding_a: float = 0.5
    companding_centering: str = "fft"
    lambda_schedule: str = "none"
    lambda_start: float = 1.0
    lambda_end: float = 0.1
    pos_emb_type: str = "APE"
    # Attention implementation used inside transformer blocks.
    # Options for self-attention: "standard", "complex", "real_valued", "phase_aware"
    # The "cross_axial" encoder family is complex-only.
    attn_type: str = "standard"
    # Base frequency for RoPE (ignored when pos_emb_type == "APE")
    rope_theta: float = 100.0
    # Randomly rotate initial 2D frequencies in Rope-Mixed (ignored otherwise)
    rope_mixed_rotate: bool = True
    # Masking strategy for vertical (column) attention in axial encoders.
    # "none"    — standard self-attention, no masking
    # "lenient" — sampled queries attend only to sampled keys;
    #             unsampled queries attend to all (sampled + unsampled)
    # "strict"  — all queries attend only to sampled keys
    # Cross-attention routing now lives in the separate "cross_axial" encoder family.
    mask_vertical_attn: str = "none"
    # FFN weight sharing across transformer layers:
    #   "none"      — every transformer layer gets its own FFN (default)
    #   "per_stage" — one shared FFN per cascade stage (all layers in that stage)
    #   "global"    — one shared FFN across the whole model (all stages; requires
    #                 every stage to use the same d_model / FFN width / dtype)
    ffn_sharing: str = "none"
    flattening_order: str = "row_major"
    # ---------------------------------------------------------------------------
    # Training hyperparameters
    # ---------------------------------------------------------------------------
    loss_mode: str = "final_only"
    # Region of k-space the reconstruction loss is computed over:
    #   "all_kspace"       — every k-space location (default)
    #   "unsampled_kspace" — only locations NOT acquired by the sampling mask; the
    #                        measured region contributes nothing to the loss.
    # "unsampled_kspace" requires a loss that operates in k-space (e.g. complex_l1 /
    # complex_l2 / complex_l2_nmse / perpendicular_loss with a k-space prediction, or
    # l1 / l2 with a kspace_companding / log_kspace norm).
    loss_function_domain: str = "all_kspace"
    # "complex_l2_nmse": per-sample sum|pred-gt|^2 / sum|gt|^2 over the loss region (scale-free complex L2).
    # "complex_l2_pointwise_normalized": mean |pred-gt|^2 / (|gt| + eps), each cell normalised by its own target.
    final_loss_type: str = "l1"
    intermediate_loss_type: str = "l1"
    # Radial frequency weighting for loss type "freq_weighted_complex_l2" (k-space only):
    #   r is the radius on the normalised square [-1, 1]^2 centred at DC, r_max = sqrt(2) (corner);
    #   w(r) = 1 + (freq_weight_m - 1) * (r / r_max)^freq_weight_gamma, then normalised to mean 1.
    #   freq_weight_m > 1 up-weights high frequencies (w = m in the corners); m == 1 == complex_l2.
    #   freq_weight_r_cap (optional): use r_max = r_cap and plateau at m for r >= r_cap, so only
    #   the region inside r_cap is de-emphasised (e.g. 0.6 = keep the central 60% box uniform).
    freq_weight_m: float = 5.0
    freq_weight_gamma: float = 1.0
    freq_weight_r_cap: Optional[float] = None
    perpendicular_mag_weighting: bool = False
    perpendicular_mag_weight_m: float = 1.0
    perpendicular_mag_weight_k: float = 0.103
    perpendicular_mag_weight_p: float = 67.0
    perpendicular_mag_weight_m_schedule: str = "none"
    perpendicular_mag_weight_m_start: float = 1.0
    perpendicular_mag_weight_m_end: float = 1.0
    # LORAKS C-matrix low-rank penalty (loss types "complex_l2_loraks" / "loraks_c"):
    #   loraks_weight    — lambda on the low-rank term added to complex_l2
    #   loraks_radius    — k-space neighbourhood radius R (Nr = 13 / 29 / 49 for R = 2 / 3 / 4)
    #   loraks_rank      — truncation rank r_C; None -> Nr // 2 (best in the notebook sweep)
    #   loraks_normalize — "ratio" (tail energy / total energy, scale-free) or "mean"
    loraks_weight: float = 0.05
    loraks_radius: int = 3
    loraks_rank: Optional[int] = None
    loraks_normalize: str = "ratio"
    epochs: int = 400
    batch_size: int = 32
    auto_batch_size: bool = True
    batch_size_search_start: int = 128
    batch_size_probe_steps: int = 3
    optimizer_type: str = "adam"
    scheduler_type: str = "cosine"
    lr: float = 1e-4
    lr_step_size: int = 40
    lr_gamma: float = 0.1
    weight_decay: float = 1e-5
    num_workers: int = 4
    grad_clip: Optional[float] = 1.0
    checkpoint_metric: str = "psnr"
