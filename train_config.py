
# ---------------------------------------------------------------------------
# fastMRI single-coil knee, random R=4 (8% ACS), 100 epochs: 4-stage multi-scale axial cascade with a RADIAL pass.
#   stage 1-4 axial           multi-scale axial, three passes per stage:
#                               rows    -> whole-row tokens
#                               columns -> whole-column tokens
#                               spokes  -> 320 half-spoke tokens from DC (160 lines through DC, split at DC into their
#                                          two halves), 160 samples each at 1 px out to the inscribed circle, periodic
#                                          RoPE over the angle; added as a RESIDUAL to the column-pass k-space
#                             d_model 320, both Q/K conv scales 1 & 3
#   data consistency: one learned scalar lambda per stage
#   loss: complex L2 on the final output only
# ---------------------------------------------------------------------------
EXPERIMENTS = [
    {
        "prefix": "fastmri",
        "name": "random_axial4_msaxial_seg1_radial320_scalardc_l2final_r4_100ep",
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
        "axial_token_axis": "both+radial",
        "radial_num_spokes": 320,
        "radial_spacing": 1.0,
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
# Band-split cascade with the sigma schedule REVERSED: sigma 1.0 -> 0.15 along the cascade (non-increasing), so the
# early stages are almost entirely the axial expert over the whole grid and the window expert (4x4 patch tokens, radial
# 5x5 / 10x10 / 20x20 windows) takes over progressively more of the high band in the later stages. Otherwise identical
# to the split4 patch-window run (whole-column axial low expert, scalar lambda DC, final-only complex L2).
# ---------------------------------------------------------------------------
EXPERIMENTS.append({
    **{k: v for k, v in EXPERIMENTS[0].items() if not k.startswith("radial_")},
    "name": "random_split4_msaxial_seg1_winpatch4_radial5x5_sigmarev_scalardc_l2final_r4_100ep",
    "encoders": ["freq_split"] * 4,
    "axial_token_axis": "both",
    "freq_split_encoders": ("axial", "window_axial"),
    "freq_split_sigma_learned": True,
    "freq_split_sigma_schedule": "geometric",
    "freq_split_sigma_start": 1.0,          # reversed: large low band first ...
    "freq_split_sigma_end": 0.15,           # ... small low band last

    "window_patch_size": (4, 4),
    "window_height": 5,
    "window_width": 5,
    "window_shift": True,
    "window_radial_scales": (1, 2, 4),
    "window_radial_bands": (0.3, 0.6),
    "window_d_model": 64,
    "window_nhead": 4,
})
