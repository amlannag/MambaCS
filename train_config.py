"""
Experiment definitions for DcTNN training.
"""

EXPERIMENTS = [
    {
        "prefix": "fastMRI",
        "name": "fastmri_kspace_magnitude_norm_complex_l2_complex_hard_dc_400epochs",
        "dataset": "fastmri",
        "image_size": (320, 320),
        "encoders": ["axial", "axial", "axial"],
        "learning": "k_space",
        "norm": "fastmri_magnitude",
        "pos_emb_type": "Rope-Axial",
        "attn_type": "complex",
        "acceleration_factors": [4],
        "lambda_schedule": "hard",
        "loss_mode": "final_only",
        "final_loss_type": "complex_l2",
        "intermediate_loss_type": "complex_l2",
        "epochs": 400,
        "seed": 42,
    },
    # {
    #     "prefix": "fastMRI",
    #     "name": "fastmri_complex_image_magnitude_norm_complex_l2_complex_hard_dc_400epochs",
    #     "dataset": "fastmri",
    #     "image_size": (320, 320),
    #     "encoders": ["axial", "axial", "axial"],
    #     "learning": "complex_image",
    #     "norm": "fastmri_magnitude",
    #     "pos_emb_type": "Rope-Axial",
    #     "attn_type": "complex",
    #     "acceleration_factors": [4],
    #     "lambda_schedule": "hard",
    #     "loss_mode": "final_only",
    #     "final_loss_type": "complex_l2",
    #     "intermediate_loss_type": "complex_l2",
    #     "epochs": 400,
    #     "seed": 42,
    # },
]
