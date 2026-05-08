"""
Experiment definitions for DcTNN training.

Each entry in EXPERIMENTS is a dict of Config field overrides.
Any key not listed falls back to the Config class default (defined in config.py).

Run a specific experiment:
    python train.py --exp_idx <N>

Submit all experiments as separate SLURM jobs:
    bash launch.sh
"""

EXPERIMENTS = [
    {
        "prefix": "KSpace",
        "name": "patch",
        "pos_emb_type": "APE",
        "encoders": ["patch", "patch", "patch"],
        "k_space_learning": True,
    },
    {
        "prefix": "KSpace",
        "name": "kaleidoscope",
        "pos_emb_type": "APE",
        "encoders": ["kaleidoscope", "kaleidoscope", "kaleidoscope"],
        "k_space_learning": True,
    },
    {
        "prefix": "KSpace",
        "name": "axial",
        "pos_emb_type": "APE",
        "encoders": ["axial", "axial", "axial"],
        "k_space_learning": True,
    },
    {
        "prefix": "Lambda_Schedule",
        "name": "patch_cosine",
        "pos_emb_type": "APE",
        "encoders": ["patch", "patch", "patch"],
        "lambda_schedule": "cosine",
        "lambda_start": 1.0,
        "lambda_end": 0.01,
        "epochs": 800,
    },
]
