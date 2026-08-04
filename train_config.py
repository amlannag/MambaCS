"""
Experiment definitions for DcTNN training.
"""

_BASE = {
    "prefix": "OASIS",
    "dataset": "oasis",
    "image_size": (256, 256),
    "encoders": ["axial", "axial", "axial"],
    "learning": "k_space",
    "norm": "none",
    "pos_emb_type": "Rope-Axial",
    "attn_type": "complex",
    "acceleration_factors": [4],
    "loss_mode": "final_only",
    "final_loss_type": "l1",
}

EXPERIMENTS = [
    # lambda=0: no soft DC, hard replacement at measured points
    {**_BASE, "name": "lambda_zero_4x", "lambda_schedule": "constant", "lambda_start": 0.0},
    # lambda=1: fixed soft DC weight of 1
    {**_BASE, "name": "lambda_one_4x", "lambda_schedule": "constant", "lambda_start": 1.0},
    # learnable lambda: per-stage nn.Parameter initialised to 0.5
    {**_BASE, "name": "lambda_learned_4x", "lambda_schedule": "none"},
    # scheduled lambda: linear ramp from 0 to 1 over training
    {**_BASE, "name": "lambda_scheduled_0to1_4x", "lambda_schedule": "linear", "lambda_start": 0.0, "lambda_end": 1.0},
]
