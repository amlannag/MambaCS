"""
Experiment definitions for DcTNN training.
"""

EXPERIMENTS = [
    {
        "prefix": "OASIS",
        "name": "image_intermediate_l1_4x",
        "dataset": "oasis",
        "image_size": (256, 256),
        "encoders": ["axial", "axial", "axial"],
        "learning": "image",
        "norm": "none",
        "pos_emb_type": "Rope-Axial",
        "attn_type": "standard",
        "acceleration_factors": [4],
        "lambda_schedule": "constant",
        "lambda_start": 0.0,
        "lambda_end": 0.0,
        "loss_mode": "intermediate_unweighted",
        "final_loss_type": "l1",
        "intermediate_loss_type": "l1",
        "seed": 42,
    },
]
