"""
Experiment definitions for DcTNN training.
"""

EXPERIMENTS = [
    {
        "prefix": "OASIS",
        "name": "lambda_cosine_0to1_l1_4x",
        "dataset": "oasis",
        "image_size": (256, 256),
        "encoders": ["axial", "axial", "axial"],
        "learning": "k_space",
        "norm": "none",
        "pos_emb_type": "Rope-Axial",
        "attn_type": "complex",
        "acceleration_factors": [4],
        "lambda_schedule": "cosine",
        "lambda_start": 0.0,
        "lambda_end": 1.0,
        "loss_mode": "final_only",
        "final_loss_type": "l1",
        "intermediate_loss_type": "l1",
    },
    {
        "prefix": "OASIS",
        "name": "lambda_cosine_1to0_l1_4x",
        "dataset": "oasis",
        "image_size": (256, 256),
        "encoders": ["axial", "axial", "axial"],
        "learning": "k_space",
        "norm": "none",
        "pos_emb_type": "Rope-Axial",
        "attn_type": "complex",
        "acceleration_factors": [4],
        "lambda_schedule": "cosine",
        "lambda_start": 1.0,
        "lambda_end": 0.0,
        "loss_mode": "final_only",
        "final_loss_type": "l1",
        "intermediate_loss_type": "l1",
    },
]
