
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

# ---------------------------------------------------------------------------
# Band-split cascade with a redesigned high-frequency expert: same axial low expert as experiment 0, high expert =
# window attention over ISOTROPIC 4x4 k-space patch tokens (80x80 token grid, no vertical bias) with RADIAL window
# sizes: 5x5-patch windows (20x20 k-space) for |k| < 0.3, 10x10 patches to 0.6, 20x20 patches (80x80 k-space) beyond.
# ---------------------------------------------------------------------------
EXPERIMENTS.append({
    **EXPERIMENTS[0],
    "name": "random_split4_msaxial_seg1_winpatch4_radial5x5_scalardc_l2final_r4_100ep",
    "encoders": ["freq_split"] * 4,
    "freq_split_encoders": ("axial", "window_axial"),
    "freq_split_sigma_learned": True,
    "freq_split_sigma_schedule": "geometric",
    "freq_split_sigma_start": 0.15,
    "freq_split_sigma_end": 1.0,

    "window_patch_size": (4, 4),
    "window_height": 5,
    "window_width": 5,
    "window_shift": True,
    "window_radial_scales": (1, 2, 4),
    "window_radial_bands": (0.3, 0.6),
    "window_d_model": 64,
    "window_nhead": 4,
})
