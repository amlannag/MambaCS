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
from typing import List, Optional


@dataclass
class Config:

    # ---------------------------------------------------------------------------
    # Experiment identity
    # ---------------------------------------------------------------------------

    # Short label for the project / paper section (e.g. "ablation", "dctnn")
    prefix: str = "MambaCS"

    # Descriptive name for this specific run (e.g. "R4only", "lr1e3", "baseline")
    name: str = "8x_acceleration"

    # Root directory where all experiment folders are created
    output_dir: str = "../Experiments"

    # Set to a checkpoint path (e.g. "../Experiments/dctnn_baseline/latest.pth")
    # to resume a stopped run; leave as None to start fresh
    resume: Optional[str] = None

    # ---------------------------------------------------------------------------
    # Data
    # ---------------------------------------------------------------------------

    # Folder of fully sampled MRI images (PNG / TIFF / NPY / NIfTI)
    data_dir: str = "/scratch/user/uqanag/OASIS/keras_png_slices_train"

    # Folder containing mask_R4.png, mask_R6.png, mask_R8.png
    mask_dir: str = "masks"

    # All images are resized to image_size x image_size
    image_size: int = 320

    # 1 for greyscale MRI, 2 for complex (real + imag channels)
    num_channels: int = 1

    # Which acceleration factors to randomly draw from during training
    # Available masks must exist in mask_dir for each value listed here
    acceleration_factors: List[int] = field(default_factory=lambda: [8])

    # Fraction of the dataset held out for validation
    val_fraction: float = 0.1
    seed: int = 42

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

    patch_size: int = 16
    nhead_patch: int = 8
    nhead_axial: int = 8
    layer_no: int = 1
    num_encoder_layers: int = 2
    learned_lambda: bool = True

    # Positional embedding type: "APE" | "Rope-Axial" | "Rope-Mixed"
    pos_emb_type: str = "APE"
    # Base frequency for RoPE (ignored when pos_emb_type == "APE")
    rope_theta: float = 100.0
    # Randomly rotate initial 2D frequencies in Rope-Mixed (ignored otherwise)
    rope_mixed_rotate: bool = True

    # ---------------------------------------------------------------------------
    # Training hyperparameters
    # ---------------------------------------------------------------------------

    epochs: int = 100

    batch_size: int = 1

    lr: float = 1e-4
    weight_decay: float = 1e-5
    num_workers: int = 4
    grad_clip: float = 1.0
