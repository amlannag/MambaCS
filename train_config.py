"""
Experiment definitions for DcTNN training.
"""

EXPERIMENTS = [
    {
        "prefix": "fastMRI",
        "name": "fastmri_complex_image_magnitude_norm_perpendicular_complex_hard_dc_400epochs",
        "dataset": "fastmri",
        "image_size": (320, 320),
        "encoders": ["axial", "axial", "axial"],
        "learning": "complex_image",
        "norm": "fastmri_magnitude",
        "pos_emb_type": "Rope-Axial",
        "attn_type": "complex",
        "acceleration_factors": [4],
        "lambda_schedule": "hard",
        "loss_mode": "final_only",
        "final_loss_type": "perpendicular_loss",
        "intermediate_loss_type": "perpendicular_loss",
        "epochs": 400,
        "seed": 42,
    }
]
