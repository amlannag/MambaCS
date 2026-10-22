
# ---------------------------------------------------------------------------
# fastMRI single-coil knee, random R=4 (8% ACS), 100 epochs: plain 4-stage multi-scale axial cascade.
#   stage 1-4 axial           multi-scale axial: whole-column tokens, both axes, Q/K conv scales 1 & 3, d_model 320
#   data consistency: one learned scalar lambda per stage
#   loss: complex L2 on the final output only
# ---------------------------------------------------------------------------
EXPERIMENTS = [
    {
        "prefix": "fastmri",
        "name": "random_axial4_msaxial_seg1_scalardc_l2final_r4_100ep",
        "dataset": "fastmri",
        "num_channels": 1,
        "image_size": (320, 320),
        "mask_type": "random",
        "acceleration_factors": [4],
        "center_fractions": [0.08],
        "learning": "k_space",
        "norm": "fastmri_magnitude",

        "encoders": ["axial"] * 4,
        "axial_d_model": 320,
        "axial_token_axis": "both",
        "axial_col_segments": 1,              # whole-column tokens
        "attn_type": "complex_ms",
        "attn_scales": (1, 3),
        "sdpa_dtype": "bfloat16",             # fused flash attention on ROCm (fp32 SDPA has no fused kernel there)
        "ffn_sharing": "global",

        "lambda_schedule": "none",
        "learned_lambda": True,
        "dc_type": "scalar",

        "final_loss_type": "complex_l2",
        "loss_mode": "final_only",

        "epochs": 100,
        "num_gpus": 1,
        "batch_size_search_start": 128,
    },
]
