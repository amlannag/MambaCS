"""
Config dataclass with default values for all hyperparameters.

To run experiments, define overrides in train_config.py and run:
    python train.py --exp_idx <N>

Encoder options for `encoders`:
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

    # Folder of training .h5 MRI files (fastMRI format, one file per scan)
    data_dir: str = "/scratch/user/uqanag/fastmri/singlecoil_train"

    # Folder of validation .h5 MRI files. When set, validation is loaded
    # directly from this directory instead of splitting data_dir.
    val_data_dir: Optional[str] = "/scratch/user/uqanag/fastmri/singlecoil_val"

    # HDF5 dataset key for raw k-space within each .h5 file
    kspace_key: str = "kspace"

    # All images are resized to image_size x image_size
    image_size: Tuple[int, int] = (640, 320)

    # Number of complex-valued image/k-space channels. Keep this at 1 for the
    # current complex-valued pipeline.
    num_channels: int = 1

    # Which acceleration factors to randomly draw from during training
    acceleration_factors: List[int] = field(default_factory=lambda: [8])

    # Legacy split controls. Ignored when val_data_dir is set.
    val_fraction: float = 0.1
    seed: int = 42
    # Cap validation set size. None = use full val set.
    max_val_samples: int = 1000

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
    encoders: List[str] = field(default_factory=lambda: ["patch", "patch", "patch"])

    patch_size: tuple = (16, 32)
    axial_row_stride: int = 2
    nhead_patch: int = 8
    nhead_axial: int = 8
    layer_no: int = 1
    num_encoder_layers: int = 2
    learned_lambda: bool = True
    # Domain the model operates in: "k_space" or "image"
    learning: str = "k_space"

    lambda_schedule: str = "none"
    lambda_start: float = 1.0
    lambda_end: float = 0.1


    pos_emb_type: str = "APE"
    # Attention implementation used inside transformer blocks.
    # Options: "standard", "complex", "real_valued", "phase_aware"
    attn_type: str = "complex"
    # Base frequency for RoPE (ignored when pos_emb_type == "APE")
    rope_theta: float = 100.0
    # Randomly rotate initial 2D frequencies in Rope-Mixed (ignored otherwise)
    rope_mixed_rotate: bool = True

    # ---------------------------------------------------------------------------
    # Training hyperparameters
    # ---------------------------------------------------------------------------

    epochs: int = 400
    batch_size: int = 16
    lr: float = 1e-4
    weight_decay: float = 1e-5
    num_workers: int = 4
    grad_clip: float = 1.0
