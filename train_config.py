"""
Experiment definitions for DcTNN training.
"""

EXPERIMENTS = [
    {
        "prefix": "fastMRI",
        "name": "phase_aware_kspace_none_perpendicular_loss_weighted_m1_to_m4_linear_4x",
        "dataset": "fastmri",
        "image_size": (320, 320),
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
        "perpendicular_mag_weighting": False,
        "perpendicular_mag_weight_m": 1.0,
        "perpendicular_mag_weight_k": 0.103,
        "perpendicular_mag_weight_p": 67.0,
        "perpendicular_mag_weight_m_schedule": "linear",
        "perpendicular_mag_weight_m_start": 1.0,
        "perpendicular_mag_weight_m_end": 4.0,
        "epochs": 400,
        "seed": 42,
    }
]
