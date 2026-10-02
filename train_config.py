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


# ReconFormer-style multi-scale attention (attn_type="complex_ms"): the 8 heads are split into
# 4 pointwise heads (kernel 1) and 4 heads whose Q/K are a complex conv over the 3-token
# neighbourhood (adjacent k-space rows/columns for axial tokens, 3x3 patch neighbourhood for
# patch tokens). V and the output projection stay pointwise; RoPE is applied after the conv.
_MULTISCALE = {
    **_BASE,
    "prefix": "multiscale",
    "attn_type": "complex_ms",
    "attn_scales": (1, 3),
}

EXPERIMENTS = [
    # Scale sweep on the axial encoder, keeping 4 heads per scale. d_model=320 so head_dim shrinks:
    # 16 heads -> head_dim 20 (10 RoPE freqs/axis), 20 heads -> head_dim 16 (8 RoPE freqs/axis).
    {
        **_MULTISCALE,
        "name": "axial_ms1357_h16_p95_slice_l2_final_r4",
        "encoders": ["axial", "axial", "axial"],
        "nhead_axial": 16,
        "attn_scales": (1, 3, 5, 7),
    },
    {
        **_MULTISCALE,
        "name": "axial_ms13579_h20_p95_slice_l2_final_r4",
        "encoders": ["axial", "axial", "axial"],
        "nhead_axial": 20,
        "attn_scales": (1, 3, 5, 7, 9),
    },
]
