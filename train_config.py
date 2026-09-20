"""
Experiment definitions for DcTNN training.
"""

# Shared recipe: k-space learning, fastMRI R=4, complex axial encoders, learned lambda DC after every
# stage, final-only loss over the whole k-space. FNet stages keep their own FFNs (excluded from sharing).
_BASE = {
    "hpc_backend": "amd",
    "model_type": "dctnn",
    "resume": None,
    "flattening_order": "row_major",
    "dataset": "fastmri",
    "image_size": (320, 320),
    "learning": "k_space",
    "norm": "fastmri_magnitude",
    "kspace_fill": None,
    "acceleration_factors": [4],
    "center_fractions": [0.08],
    "mask_type": "random",
    "axial_row_stride": 1,
    "mask_vertical_attn": "none",
    "layer_no": 1,
    "num_encoder_layers": 2,
    "nhead_axial": 8,
    "layer_norm_eps": 1e-5,
    "attn_type": "complex",
    "pos_emb_type": "Rope-Axial",
    "rope_theta": 100.0,
    "lambda_schedule": "none",
    "learned_lambda": True,
    "loss_mode": "final_only",
    "loss_function_domain": "all_kspace",
    "epochs": 100,
    "batch_size": 32,
    "auto_batch_size": True,
    "batch_size_search_start": 128,
    "lr": 2e-4,
    "ffn_sharing": "global",
}

# FNet cascade sweep: frequency-weighted complex L2 (m=3, g=0.5, plateau at r=0.6) on the final output.
_FNET_BASE = {
    **_BASE,
    "prefix": "fnet_cascade",
    "fnet_fft_norm": "ortho",
    "fnet_with_embedding": True,
    "final_loss_type": "freq_weighted_complex_l2",
    "intermediate_loss_type": "freq_weighted_complex_l2",
    "freq_weight_m": 3.0,
    "freq_weight_gamma": 0.5,
    "freq_weight_r_cap": 0.6,
}

EXPERIMENTS = [
    # Exp 1: FNet (horizontal + vertical tokens) -> cross-attention -> 3 axial
    {
        **_FNET_BASE,
        "name": "dctnn_fnet_both_cross_axial_3axial_fastmri_mag_learned_lambda_freqw_l2_m3_g0.5_cap0.6_final_r4",
        "encoders": ["fnet", "cross_axial", "axial", "axial", "axial"],
        "fnet_token_axis": "both",
    },

    # Exp 2: FNet (vertical / column tokens) -> cross-attention -> 3 axial
    {
        **_FNET_BASE,
        "name": "dctnn_fnet_vertical_cross_axial_3axial_fastmri_mag_learned_lambda_freqw_l2_m3_g0.5_cap0.6_final_r4",
        "encoders": ["fnet", "cross_axial", "axial", "axial", "axial"],
        "fnet_token_axis": "vertical",
    },

    # Exp 3: 3 axial cascades trained with the pointwise-normalised complex L2, |pred - gt|^2 / (|gt| + eps)
    {
        **_BASE,
        "prefix": "pointwise_l2",
        "name": "dctnn_axial_3axial_fastmri_mag_learned_lambda_pointwise_norm_l2_final_r4",
        "encoders": ["axial", "axial", "axial"],
        "final_loss_type": "complex_l2_pointwise_normalized",
    },
]
