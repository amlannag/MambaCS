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
        "prefix": "PosEmbed",
        "name": "APE_baseline",
        "pos_emb_type": "APE",
        "encoders": ["patch", "patch", "patch"],
    },
    {
        "prefix": "PosEmbed",
        "name": "RopeAxial",
        "pos_emb_type": "Rope-Axial",
        "rope_theta": 100.0,
    },
    {
        "prefix": "PosEmbed",
        "name": "RopeMixed",
        "pos_emb_type": "Rope-Mixed",
        "rope_theta": 10.0,
        "rope_mixed_rotate": True,
    },
]
