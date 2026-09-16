"""
Experiment definitions for DcTNN training.
"""

EXPERIMENTS = [

    {
        "prefix": "reconformer_kspace",
        "name": "reconformer_kspace_fastmri_mag_hard_dc_r4",
        "hpc_backend": "amd",
        "model_type": "reconformer",
        "resume": None,
        "dataset": "fastmri",
        "image_size": (320, 320),
        "learning": "k_space",
        "norm": "fastmri_magnitude",
        "kspace_fill": None,
        "acceleration_factors": [4],
        "center_fractions": [0.08],
        "mask_type": "random",
        "reconformer_num_ch": (96, 48, 24),
        "reconformer_num_iter": 5,
        "reconformer_down_scales": (2.0, 1.0, 1.5),
        "reconformer_num_heads": (6, 6, 6),
        "reconformer_depths": (2, 1, 1),
        "reconformer_window_sizes": (8, 8, 8),
        "reconformer_mlp_ratio": 2.0,
        "reconformer_resi_connection": "1conv",
        "reconformer_use_checkpoint": (False, False, True, True, False, False),
        "loss_mode": "final_only",
        "final_loss_type": "complex_l2",
        "intermediate_loss_type": "complex_l2",
        "epochs": 100,
        "batch_size": 4,
        "auto_batch_size": True,
        "batch_size_search_start": 16,
        "lr": 2e-4,
    }
]
