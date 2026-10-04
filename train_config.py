"""
Experiment definitions for DcTNN training.
"""

# Shared recipe: k-space learning, fastMRI R=4, complex encoders, learned lambda DC after every
# stage, final-only complex-L2 loss over the whole k-space, per-slice p95 magnitude normalisation.
# FNet stages keep their own FFNs (excluded from sharing).
_BASE = {
    "hpc_backend": "amd",
    "model_type": "dctnn",
    "resume": None,
    "flattening_order": "row_major",
    "dataset": "fastmri",
    "image_size": (320, 320),
    "learning": "k_space",
    "norm": "fastmri_magnitude",
    "norm_scope": "slice",
    "norm_quantile": 0.95,
    "kspace_fill": None,
    "acceleration_factors": [4],
    "center_fractions": [0.08],
    "mask_type": "random",
    "axial_row_stride": 1,
    "mask_vertical_attn": "none",
    "layer_no": 1,
    "num_encoder_layers": 2,
    "nhead_axial": 8,
    "nhead_patch": 8,
    "layer_norm_eps": 1e-5,
    "pos_emb_type": "Rope-Axial",
    "rope_theta": 100.0,
    "lambda_schedule": "none",
    "learned_lambda": True,
    "loss_mode": "final_only",
    "loss_function_domain": "all_kspace",
    "final_loss_type": "complex_l2",
    "intermediate_loss_type": "complex_l2",
    "epochs": 100,
    "batch_size": 32,
    "auto_batch_size": True,
    "batch_size_search_start": 150,
    "lr": 2e-4,
    "ffn_sharing": "global",
}

# Plain complex axial attention with optional learnable k-space stems (see Config.global_filter / kspace_conv).
_KSPACE_STEM = {
    **_BASE,
    "prefix": "kspace_stem",
    "attn_type": "complex",
    "encoders": ["axial", "axial", "axial"],
    "kspace_conv_channels": 8,
    "kspace_conv_kernel": 3,
}

EXPERIMENTS = [
  
    {
        **_BASE,
        "prefix": "combined",
        "name": "fnet_both_cross_axial_ms13_gfilt_freqw_m3_g0.5_cap0.6_p95_slice_400ep_r4",
        "encoders": ["fnet", "cross_axial", "axial", "axial", "axial"],
        "fnet_token_axis": "both",
        "fnet_with_embedding": True,
        "fnet_share_ffn": True,
        "fnet_fft_norm": "ortho",
        "attn_type": "complex_ms",
        "attn_scales": (1, 3),
        "global_filter": True,
        "final_loss_type": "freq_weighted_complex_l2",
        "intermediate_loss_type": "freq_weighted_complex_l2",
        "freq_weight_m": 3.0,
        "freq_weight_gamma": 0.5,
        "freq_weight_r_cap": 0.6,
        "epochs": 400,
    },
]
