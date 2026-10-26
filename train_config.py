
# ---------------------------------------------------------------------------
# fastMRI single-coil knee, random R=4 (8% ACS), 2 GPUs x 100 epochs.
# Same cascade as the fixed-band reference run (fnet -> cross_axial -> 3x [window_axial REFINE +
# freq_split CONSOL.]) but the hard-coded radial WRITE bands of the refine stages are replaced by LEARNED
# per-pixel sigmoid write weights (init 0.5): candidate = previous_output + sigmoid(w_i) * residual, so each
# k-space pixel of each refine stage learns how much of its refine to mix with the previous layer output
# (0 = keep previous output, 1 = full refine), instead of being fixed to a radial band. The banded loss
# curriculum is unchanged (it only weights supervision, it does not restrict writes).
# ---------------------------------------------------------------------------
FASTMRI_FNET_CROSS_REFINE3_MIXW = {
    "prefix": "fastmri",
    "name": "random_fnet_cross_refine3consol3_mixw_sigmoid_bandedmean_r4_100ep",

    # data
    "dataset": "fastmri",
    "num_channels": 1,
    "image_size": (320, 320),
    "mask_type": "random",
    "acceleration_factors": [4],
    "center_fractions": [0.08],
    "learning": "k_space",
    "norm": "fastmri_magnitude",

    # cascade
    "encoders": ["fnet", "cross_axial", "window_axial", "freq_split", "window_axial", "freq_split", "window_axial", "freq_split"],
    "stage_mix_weights": [None, None, 0.5, None, 0.5, None, 0.5, None],
    "attn_type": "complex_ms",
    "attn_scales": (1, 3),
    "sdpa_dtype": "bfloat16",             # fused flash attention on ROCm (fp32 SDPA has no fused kernel there)
    "ffn_sharing": "global",

    # fnet
    "fnet_token_axis": "both",
    "fnet_with_embedding": True,

    # axial (cross_axial opener + freq_split low expert)
    "axial_d_model": 320,
    "axial_token_axis": "both",
    "axial_col_segments": 1,

    # window_axial (refine stages + freq_split high expert)
    "window_patch_size": (4, 4),
    "window_num_layers": 4,
    "window_height": 10,
    "window_width": 10,
    "window_shift": True,
    "window_radial_scales": (1, 2, 4),
    "window_radial_bands": (0.3, 0.6),
    "window_d_model": 128,
    "window_nhead": 8,

    # freq_split
    "freq_split_encoders": ("axial", "window_axial"),
    "freq_split_sigma_learned": True,
    "freq_split_sigma_schedule": "geometric",
    "freq_split_sigma_start": 1.0,
    "freq_split_sigma_end": 0.15,

    # data consistency
    "lambda_schedule": "none",
    "learned_lambda": True,
    "dc_type": "scalar",

    # loss
    "final_loss_type": "complex_l2",
    "intermediate_loss_type": "complex_l2",
    "loss_mode": "intermediate_banded",
    "banded_loss_bands": [("all",), ("all",), ("disc", 0.30), ("disc", 0.30), ("ring", 0.30, 0.67), ("disc", 0.67), ("ring", 0.67, None), ("all",)],
    "banded_loss_normalise": "mean",
    "banded_loss_floor": 0.0,

    # training
    "epochs": 100,
    "num_gpus": 2,
    "batch_size_search_start": 128,
}

# ---------------------------------------------------------------------------
# fastMRI single-coil knee, random R=4 (8% ACS), 2 GPUs x 100 epochs.
# Deeper variant of FASTMRI_FNET_CROSS_REFINE3_MIXW: two extra refine + consolidate pairs (5x instead of 3x),
# same learned per-pixel sigmoid write weights on every window_axial refine stage (init 0.5). The banded-loss
# curriculum is the 5-way generalisation of the 3-way one: refine i owns the shell disc 0.20 / ring 0.20-0.35 /
# ring 0.35-0.50 / ring 0.50-0.67 / ring 0.67+, and the following freq_split consolidate stage is held to the
# cumulative disc (0.20 / 0.35 / 0.50 / 0.67 / all). freq_split sigma schedule adapts automatically to the 5
# split stages. Batch search starts at 128 global (64/GPU) like the 3x variant; the deeper cascade will back off
# via the auto search if the GPUs cannot fit it.
# ---------------------------------------------------------------------------
FASTMRI_FNET_CROSS_REFINE5_MIXW = {
    "prefix": "fastmri",
    "name": "random_fnet_cross_refine5consol5_mixw_sigmoid_bandedmean_r4_100ep",

    # data
    "dataset": "fastmri",
    "num_channels": 1,
    "image_size": (320, 320),
    "mask_type": "random",
    "acceleration_factors": [4],
    "center_fractions": [0.08],
    "learning": "k_space",
    "norm": "fastmri_magnitude",

    # cascade
    "encoders": ["fnet", "cross_axial", "window_axial", "freq_split", "window_axial", "freq_split",
                 "window_axial", "freq_split", "window_axial", "freq_split", "window_axial", "freq_split"],
    "stage_mix_weights": [None, None, 0.5, None, 0.5, None, 0.5, None, 0.5, None, 0.5, None],
    "attn_type": "complex_ms",
    "attn_scales": (1, 3),
    "sdpa_dtype": "bfloat16",             # fused flash attention on ROCm (fp32 SDPA has no fused kernel there)
    "ffn_sharing": "global",

    # fnet
    "fnet_token_axis": "both",
    "fnet_with_embedding": True,

    # axial (cross_axial opener + freq_split low expert)
    "axial_d_model": 320,
    "axial_token_axis": "both",
    "axial_col_segments": 1,

    # window_axial (refine stages + freq_split high expert)
    "window_patch_size": (4, 4),
    "window_num_layers": 4,
    "window_height": 10,
    "window_width": 10,
    "window_shift": True,
    "window_radial_scales": (1, 2, 4),
    "window_radial_bands": (0.3, 0.6),
    "window_d_model": 128,
    "window_nhead": 8,

    # freq_split
    "freq_split_encoders": ("axial", "window_axial"),
    "freq_split_sigma_learned": True,
    "freq_split_sigma_schedule": "geometric",
    "freq_split_sigma_start": 1.0,
    "freq_split_sigma_end": 0.15,

    # data consistency
    "lambda_schedule": "none",
    "learned_lambda": True,
    "dc_type": "scalar",

    # loss
    "final_loss_type": "complex_l2",
    "intermediate_loss_type": "complex_l2",
    "loss_mode": "intermediate_banded",
    "banded_loss_bands": [("all",), ("all",),
                          ("disc", 0.20), ("disc", 0.20),
                          ("ring", 0.20, 0.35), ("disc", 0.35),
                          ("ring", 0.35, 0.50), ("disc", 0.50),
                          ("ring", 0.50, 0.67), ("disc", 0.67),
                          ("ring", 0.67, None), ("all",)],
    "banded_loss_normalise": "mean",
    "banded_loss_floor": 0.0,

    # training
    "epochs": 100,
    "num_gpus": 2,
    "batch_size_search_start": 128,
}

EXPERIMENTS = [
    FASTMRI_FNET_CROSS_REFINE3_MIXW,
    FASTMRI_FNET_CROSS_REFINE5_MIXW,
]
