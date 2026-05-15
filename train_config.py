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
        "prefix": "MixedDomain",
        "name": "kspace_img_img",
        "pos_emb_type": "APE",
        "encoders": ["axial", "patch", "patch"],
        "k_space_learning": [True, False, False],
    },
    {
        "prefix": "MixedDomain",
        "name": "img_kspace_img",
        "pos_emb_type": "APE",
        "encoders": ["patch", "axial", "patch"],
        "k_space_learning": [False, True, False],
    },
    {
        "prefix": "MixedDomain",
        "name": "img_kspace_kspace",
        "pos_emb_type": "APE",
        "encoders": ["patch", "axial", "axial"],
        "k_space_learning": [False, True, True],
    },
    {
        "prefix": "MixedDomain",
        "name": "img_axial_all",
        "pos_emb_type": "APE",
        "encoders": ["axial", "axial", "axial"],
        "k_space_learning": False,
    },
]
