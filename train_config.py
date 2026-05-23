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
    # --- RoPE positional embedding ablations ---
    {
        "prefix": "RoPE",
        "name": "kaleidoscope_rope_axial",
        "encoders": ["kaleidoscope", "kaleidoscope", "kaleidoscope"],
        "pos_emb_type": "Rope-Axial",
    },
    {
        "prefix": "RoPE",
        "name": "kaleidoscope_rope_mixed",
        "encoders": ["kaleidoscope", "kaleidoscope", "kaleidoscope"],
        "pos_emb_type": "Rope-Mixed",
    },
    {
        "prefix": "RoPE",
        "name": "axial_rope_axial",
        "encoders": ["axial", "axial", "axial"],
        "pos_emb_type": "Rope-Axial",
    },
]
