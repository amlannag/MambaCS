"""
Experiment definitions for DcTNN training.
"""

EXPERIMENTS = [
    {
        "prefix": "OASIS",
        "name": "kspace_lambda_zero_perpendicular_loss_phase_aware_4x",
        "dataset": "oasis",
        "image_size": (256, 256),
        "encoders": ["axial", "axial", "axial"],
        "learning": "k_space",
        "norm": "none",
        "pos_emb_type": "Rope-Axial",
        "attn_type": "phase_aware",
        "acceleration_factors": [4],
        "lambda_schedule": "constant",
        "lambda_start": 0.0,
        "lambda_end": 0.0,
        "loss_mode": "final_only",
        "final_loss_type": "perpendicular_loss",
        "intermediate_loss_type": "perpendicular_loss",
        "epochs": 400,
        "seed": 42,
    },
]
