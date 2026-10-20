
EXPERIMENTS = [
    # fastMRI knee, 3 axial cascades, complex-L2 final-only loss, random R=4 mask (the standard axial-3 baseline)
    # with a learned data-consistency weight for EVERY k-space pixel (one [H, W] map per stage) instead of one
    # scalar lambda per stage; every weight starts at 1.
    {
        "prefix": "fastMRI",
        "name": "axial3_l2_final_pixelwise_dc_init1_r4",
        "dataset": "fastmri",
        "encoders": ["axial", "axial", "axial"],
        "final_loss_type": "complex_l2",
        "loss_mode": "final_only",
        "mask_type": "random",
        "acceleration_factors": [4],
        "center_fractions": [0.08],
        "lambda_schedule": "none",
        "dc_type": "pixelwise",
        "dc_lambda_init": 1.0,
        "batch_size": 128,
        "batch_size_search_start": 128,
        "epochs": 100,
    },
]
