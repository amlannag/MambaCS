"""
Config dataclass with default values for all hyperparameters.

To run experiments, define overrides in train_config.py and run:
    python train.py --exp_idx <N>

Encoder options for `encoders`:
    "cross_axial"   — vertical-only sampled/unsampled complex cross-attention
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
    axial_row_stride: int = 1
    nhead_patch: int = 8
    nhead_axial: int = 8
    layer_no: int = 1
    num_encoder_layers: int = 2
    learned_lambda: bool = True
    # Domain the model operates in: "k_space", "image", or "complex_image"
    learning: str = "k_space"
    # Normalisation: "zscore", "fastmri_magnitude", "robust_shifted", "kspace_companding", "log_kspace", or None
    norm: str = "zscore"
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
    # ---------------------------------------------------------------------------
    # Training hyperparameters
    # ---------------------------------------------------------------------------
    loss_mode: str = "final_only"
    final_loss_type: str = "l1"
    intermediate_loss_type: str = "l1"
    perpendicular_mag_weighting: bool = False
    perpendicular_mag_weight_m: float = 1.0
    perpendicular_mag_weight_k: float = 0.103
    perpendicular_mag_weight_p: float = 67.0
    perpendicular_mag_weight_m_schedule: str = "none"
    perpendicular_mag_weight_m_start: float = 1.0
    perpendicular_mag_weight_m_end: float = 1.0
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
