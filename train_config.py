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
    # Scale sweep on the axial encoder, keeping 4 heads per scale. d_model=320 so head_dim shrinks:
    # 16 heads -> head_dim 20 (10 RoPE freqs/axis), 20 heads -> head_dim 16 (8 RoPE freqs/axis).
   
    # k-space stem ablation on the plain complex axial encoder (RoPE, learned lambda, p95). Both stems are
    # identities at init and sit on each encoder's input before tokenisation (GlobalFilter -> KSpaceConvStem).
    # idx 4 is the matching control with neither stem.
    {
        **_KSPACE_STEM,
        "name": "axial_complex_kconv_p95_slice_l2_final_r4",
        "kspace_conv": True,
    },
    {
        **_KSPACE_STEM,
        "name": "axial_complex_gfilt_p95_slice_l2_final_r4",
        "global_filter": True,
    },
    {
        **_KSPACE_STEM,
        "name": "axial_complex_kconv_gfilt_p95_slice_l2_final_r4",
        "kspace_conv": True,
        "global_filter": True,
    },
]
