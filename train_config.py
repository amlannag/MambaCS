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
        "prefix": "RoPE",
        "name": "patch_rope_mixed",
        "encoders": ["patch", "patch", "patch"],
        "pos_emb_type": "Rope-Mixed",
    },
]
