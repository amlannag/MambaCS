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
    "final_loss_type": "l1",
    "intermediate_loss_type": "l1",
}

EXPERIMENTS = [
    # linear lambda schedule 0→1 with intermediate loss at every encoder stage
    {
        **_BASE,
        "name": "lambda_scheduled_intermediate_l1_4x",
        "lambda_schedule": "linear",
        "lambda_start": 0.0,
        "lambda_end": 1.0,
        "loss_mode": "intermediate_unweighted",
    },
]
