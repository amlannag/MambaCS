"""
Experiment definitions for DcTNN training.
"""

EXPERIMENTS = [
    {
        "prefix": "fastMRI",
        "name": "dctnn_complex_phase_aware_r4_100epochs",
        "hpc_backend": "amd",
        "model_type": "dctnn",
        "dataset": "fastmri",
        "image_size": (320, 320),
        "learning": "complex_image",
        "norm": "fastmri_magnitude",
        "acceleration_factors": [4],
        "center_fractions": [0.08],
        "mask_type": "random",
        "encoders": ["patch", "patch", "patch"],
        "attn_type": "phase_aware",
        "loss_mode": "final_only",
        "final_loss_type": "complex_l1",
        "intermediate_loss_type": "complex_l1",
        "epochs": 100,
        "batch_size": 32,
        "auto_batch_size": True,
        "batch_size_search_start": 128,
        "lr": 2e-4,
    },
]
