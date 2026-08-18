"""
Experiment definitions for DcTNN training.
"""

EXPERIMENTS = [
    {
        "prefix": "fastMRI",
        "name": "complex_fastmri_kspace_none_4x",
        "dataset": "fastmri",
        "image_size": (320, 320),
        "encoders": ["axial", "axial", "axial"],
        "learning": "k_space",
        "norm": "none",
        "pos_emb_type": "Rope-Axial",
        "attn_type": "complex",
        "acceleration_factors": [4],
        "lambda_schedule": "constant",
        "lambda_start": 0.0,
        "lambda_end": 0.0,
        "loss_mode": "final_only",
        "final_loss_type": "perpendicular_loss",
        "intermediate_loss_type": "perpendicular_loss",
        "epochs": 100,
        "seed": 42,
    },
]
