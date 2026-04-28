"""
Central configuration for DcTNN training.

Edit this file, then run:  python train.py

Outputs are saved to:
    {output_dir}/{prefix}_{name}/
        config.json        — copy of this config
        best_model.pth     — weights with lowest validation loss
        latest.pth         — most recent checkpoint (use for resuming)
        metrics.json       — per-epoch train/val metrics

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
    prefix: str = "PatchOnly"

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

    # ---------------------------------------------------------------------------
    # Training hyperparameters
    # ---------------------------------------------------------------------------

    epochs: int = 100

    batch_size: int = 1

    lr: float = 1e-4
    weight_decay: float = 1e-5
    num_workers: int = 4
    grad_clip: float = 1.0


# Singleton instance — train.py does:  from train_config import cfg
cfg = Config()
