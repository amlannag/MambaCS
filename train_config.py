"""
Central configuration for DcTNN training.

Edit this file, then run:  python train.py

Outputs are saved to:
    {experiment.output_dir}/{experiment.prefix}_{experiment.name}/
        config.json        — copy of this config
        best_model.pth     — weights with lowest validation loss
        latest.pth         — most recent checkpoint (use for resuming)
        metrics.json       — per-epoch train/val metrics
"""

from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Experiment identity
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    # Short label for the project / paper section (e.g. "ablation", "dctnn")
    prefix: str = "dctnn"

    # Descriptive name for this specific run (e.g. "R4only", "lr1e3", "baseline")
    name: str = "baseline"

    # Root directory where all experiment folders are created
    output_dir: str = "../Experiments"

    # Set to a checkpoint path (e.g. "experiments/dctnn_baseline/latest.pth")
    # to resume a stopped run; leave as None to start fresh
    resume: Optional[str] = None


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
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
    acceleration_factors: List[int] = field(default_factory=lambda: [4, 6, 8])

    # Fraction of the dataset held out for validation
    val_fraction: float = 0.1

    # Seed used for the train / val split (keeps the split reproducible)
    seed: int = 42


# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    # Size of each image patch / Kaleidoscope token (must divide image_size)
    patch_size: int = 16
    # Attention heads in the patch and Kaleidoscope transformers
    nhead_patch: int = 8
    # Attention heads in the axial transformer
    nhead_axial: int = 8
    # Number of cascaded denoising blocks inside each TNN module
    layer_no: int = 1
    # Transformer encoder layers per TNN module
    num_encoder_layers: int = 2
    # If True, the data-consistency weighting lambda is a learned parameter
    learned_lambda: bool = True


# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    epochs: int = 100

    # batch_size > 1 is supported; all samples in a batch share one random mask
    batch_size: int = 1

    # Peak learning rate for Adam
    lr: float = 1e-4

    weight_decay: float = 1e-5

    # DataLoader worker processes (set to 0 on Windows or for debugging)
    num_workers: int = 4

    # Max gradient norm for clipping (helps transformer training stability)
    grad_clip: float = 1.0


# ---------------------------------------------------------------------------
# Top-level config  (import this in train.py)
# ---------------------------------------------------------------------------

@dataclass
class Config:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    data:       DataConfig       = field(default_factory=DataConfig)
    model:      ModelConfig      = field(default_factory=ModelConfig)
    training:   TrainingConfig   = field(default_factory=TrainingConfig)


# Singleton instance — train.py does:  from train_config import cfg
cfg = Config()
