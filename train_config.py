"""
Experiment definitions for ReconFormer training.
"""

EXPERIMENTS = [
    {
        "prefix": "fastMRI",
        "name": "reconformer_random_mask_r4_50epochs",
        "hpc_backend": "amd",
        "model_type": "reconformer",
        "dataset": "fastmri",
        "image_size": (320, 320),
        "learning": "complex_image",
        "norm": "reconformer",
        "acceleration_factors": [4],
        "center_fractions": [0.08],
        "mask_type": "random",
        "loss_mode": "final_only",
        "final_loss_type": "complex_l1",
        "intermediate_loss_type": "complex_l1",
        "epochs": 50,
        "batch_size": 16,
        "auto_batch_size": True,
        "batch_size_search_start": 32,
        "lr": 2e-4,
    },
]
