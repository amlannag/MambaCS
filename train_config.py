# ---------------------------------------------------------------------------
# fastMRI brain multicoil, equispaced R=4 (8% ACS), 16 virtual coils (SVD coil compression), 400 epochs.
# coil_compress keeps the top-16 PCA virtual coils in energy order: 16-coil volumes keep all 16 (full rank, a lossless
# change of basis), 20-coil volumes keep the top 16 of 20 (volumes with < 16 coils are zero-padded to 16 channels).
# Same recipe as the 8-coil 400-epoch run (brain_..._400ep_fnetnbrinit): radial pass on the freq_split low expert,
# pixel-wise DC (dc_type='pixelwise'), learned per-pixel sigmoid write filters on the refine stages.
# Trained from scratch: every stage has coil-count-dependent weights (embeddings 8*320 -> 16*320 wide, coil
# embeddings 8 -> 16), so the 8-coil checkpoints cannot be loaded. Fresh 400-epoch cosine LR schedule from lr=1e-4.
# 16 coils = 2x the tokens of the 8-coil run, so the batch search starts at half its size.
# ---------------------------------------------------------------------------
BRAIN_FNET_CROSS_REFINE3_MIXW_PIXDC_RADIAL_16VC_400EP = {
    "prefix": "brain",
    "name": "equi_fnet_nbr_percoil_refine3consol3_mixw_pixdc_radial_bandedmean_r4_400ep_16vc",

    # data
    "dataset": "fastmri_brain",
    "num_channels": 16,
    "image_size": (320, 320),
    "mask_type": "equispaced",
    "acceleration_factors": [4],
    "center_fractions": [0.08],
    "learning": "k_space",
    "norm": "fastmri_magnitude",

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
    "epochs": 400,
    "num_gpus": 2,
    "batch_size_search_start": 32,   # half the 8-coil run's 64: 16 coils = 2x the tokens
    "lr": 1e-4,
}

EXPERIMENTS = [
    BRAIN_FNET_CROSS_REFINE3_MIXW_PIXDC_RADIAL_16VC_400EP,
]
