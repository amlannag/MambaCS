
EXPERIMENTS = [
    {
        "prefix": "fastmri",
        "name": "axial_vertical3_singlecoil_complex_l2_learned_lambda_r4",
        "dataset": "fastmri",
        "num_channels": 1,
        "axial_d_model": 320,
        "batch_size": 32,
        "batch_size_search_start": 128,
        "encoders": ["axial", "axial", "axial"],
        "axial_token_axis": "vertical",
        "attn_type": "complex_ms",
        "attn_scales": (1, 3),
        "lambda_schedule": "none",
        "ffn_sharing": "global",
        "global_filter": False,
        "global_filter_mid": False,
        "kspace_conv": False,
        "image_conv": False,
        "final_loss_type": "complex_l2",
        "intermediate_loss_type": "complex_l2",
    },
]
