# ---------------------------------------------------------------------------
# fastMRI single-coil knee, random R=4 (8% ACS), 2 GPUs x 100 epochs.
# Combines all three learned-filter mechanisms on the 8-stage cascade:
#   * RADIAL pass on the freq_split low expert (axial_token_axis='both+radial'),
#   * LEARNED PER-PIXEL data-consistency lambdas (dc_type='pixelwise': one softplus-constrained lambda per
#     k-space pixel and stage, init 0.5),
#   * LEARNED per-pixel sigmoid WRITE filters on the refine stages (stage_mix_weights, init 0.5).
# The banded-loss curriculum matches the fixed-band reference run.
# Warm start: stages 0 (fnet) and 1 (cross_axial) are copied from the trained fixed-band freqsplit run and
# stay trainable. Batch search starts at 192 global (96/GPU on 2 GPUs) and backs off via the auto search if
# it does not fit.
# ---------------------------------------------------------------------------
FASTMRI_FNET_CROSS_REFINE3_MIXW_PIXDC_RADIAL = {
    "prefix": "fastmri",
    "name": "random_fnet_cross_refine3consol3_mixw_pixdc_radial_bandedmean_r4_100ep_fnetnbrinit",

    # data
    "dataset": "fastmri",
    "num_channels": 1,
    "image_size": (320, 320),
    "mask_type": "random",
    "acceleration_factors": [4],
    "center_fractions": [0.08],
    "learning": "k_space",
    "norm": "fastmri_magnitude",

    # warm start — fnet + cross_axial openers from the trained fixed-band freqsplit run
    # (experiments are saved flat as ../Experiments/<prefix>_<name>/ on HPC)
    "init_stages_from": "../Experiments/fastmri_random_fnet_cross_refine3consol3_win_d128_split_sigmarev_bandedmean_r4_100ep/best_model.pth",
    "init_stages": [0, 1],

    # cascade
    "encoders": ["fnet", "cross_axial", "window_axial", "freq_split", "window_axial", "freq_split", "window_axial", "freq_split"],
    "stage_mix_weights": [None, None, 0.5, None, 0.5, None, 0.5, None],
    "attn_type": "complex_ms",
    "attn_scales": (1, 3),
    "sdpa_dtype": "bfloat16",             # fused flash attention on ROCm (fp32 SDPA has no fused kernel there)
    "ffn_sharing": "global",

    "fnet_token_axis": "both",
    "fnet_with_embedding": True,

    # axial (cross_axial opener + freq_split low expert) — radial spoke pass ON
    "axial_d_model": 320,
    "axial_token_axis": "both+radial",
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

    # data consistency — one learned lambda PER K-SPACE PIXEL per stage
    "lambda_schedule": "none",
    "learned_lambda": True,
    "dc_type": "pixelwise",
    "dc_lambda_init": 0.5,
    "final_loss_type": "complex_l2",
    "intermediate_loss_type": "complex_l2",
    "loss_mode": "intermediate_banded",
    "banded_loss_bands": [("all",), ("all",), ("disc", 0.30), ("disc", 0.30), ("ring", 0.30, 0.67), ("disc", 0.67), ("ring", 0.67, None), ("all",)],
    "banded_loss_normalise": "mean",
    "banded_loss_floor": 0.0,

    "epochs": 100,
    "num_gpus": 2,
    "batch_size_search_start": 192,
}

# ---------------------------------------------------------------------------
# fastMRI brain multicoil, equispaced R=4 (8% ACS), 8 virtual coils (SVD coil compression), 2 GPUs x 100 epochs.
# Same learned-filter recipe as the knee MIXW_PIXDC_RADIAL run, on the multicoil cascade:
#   * openers: fnet -> neighbourhood (learned-GRAPPA cross-attention) -> cross_axial_per_coil,
#   * 3x [window_axial REFINE + freq_split CONSOL] with LEARNED per-pixel sigmoid WRITE filters on the refine
#     stages (stage_mix_weights, init 0.5),
#   * RADIAL pass on the freq_split low expert (axial_token_axis='both+radial'),
#   * LEARNED PER-PIXEL data-consistency lambdas (dc_type='pixelwise', init 0.5).
# Warm start: stages 0 (fnet), 1 (neighbourhood) and 2 (cross_axial_per_coil) are copied from the trained
# brain run ..._fnetnbrinit and stay trainable. Batch search starts at 32 global (16/GPU on 2 GPUs): 8 coils =
# 8x the tokens of the knee runs.
# ---------------------------------------------------------------------------
BRAIN_FNET_CROSS_REFINE3_MIXW_PIXDC_RADIAL = {
    "prefix": "brain",
    "name": "equi_fnet_nbr_percoil_refine3consol3_mixw_pixdc_radial_bandedmean_r4_100ep_fnetnbrinit",

    # data
    "dataset": "fastmri_brain",
    "num_channels": 8,
    "image_size": (320, 320),
    "mask_type": "equispaced",
    "acceleration_factors": [4],
    "center_fractions": [0.08],
    "learning": "k_space",
    "norm": "fastmri_magnitude",

    # warm start — fnet + neighbourhood + cross_axial_per_coil openers from the trained brain run
    "init_stages_from": "../Experiments/brain_equi_fnet_nbr_percoil_refine3consol3_win_d128_split_sigmarev_bandedmean_r4_100ep_fnetnbrinit/best_model.pth",
    "init_stages": [0, 1, 2],

    # cascade
    "encoders": ["fnet", "neighbourhood", "cross_axial_per_coil", "window_axial", "freq_split", "window_axial", "freq_split", "window_axial", "freq_split"],
    "stage_mix_weights": [None, None, None, 0.5, None, 0.5, None, 0.5, None],
    "attn_type": "complex_ms",
    "attn_scales": (1, 3),
    "sdpa_dtype": "bfloat16",             # fused flash attention on ROCm (fp32 SDPA has no fused kernel there)
    "ffn_sharing": "global",

    # fnet
    "fnet_token_axis": "both",
    "fnet_with_embedding": True,

    # neighbourhood
    "neighbourhood_strip_len": 40,
    "neighbourhood_n_neighbours": 2,
    "neighbourhood_kx_halo": 0,
    "neighbourhood_attn_type": "complex",

    # axial (cross_axial_per_coil opener + freq_split low expert) — radial spoke pass ON
    "axial_d_model": 320,
    "axial_token_axis": "both+radial",
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

    # data consistency — one learned lambda PER K-SPACE PIXEL per stage
    "lambda_schedule": "none",
    "learned_lambda": True,
    "dc_type": "pixelwise",
    "dc_lambda_init": 0.5,

    # loss
    "final_loss_type": "complex_l2",
    "intermediate_loss_type": "complex_l2",
    "loss_mode": "intermediate_banded",
    "banded_loss_bands": [("all",), ("all",), ("all",), ("disc", 0.30), ("disc", 0.30), ("ring", 0.30, 0.67), ("disc", 0.67), ("ring", 0.67, None), ("all",)],
    "banded_loss_normalise": "mean",
    "banded_loss_floor": 0.0,

    # training
    "epochs": 100,
    "num_gpus": 2,
    "batch_size_search_start": 32,
}

EXPERIMENTS = [
    FASTMRI_FNET_CROSS_REFINE3_MIXW_PIXDC_RADIAL,
    BRAIN_FNET_CROSS_REFINE3_MIXW_PIXDC_RADIAL,
]