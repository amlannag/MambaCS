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
    "batch_size_search_start": 250,
    "lr": 2e-4,
    "ffn_sharing": "global",
}


# Normalisation ablation on the plain 3x axial encoder (complex axial attention, learned-lambda DC after every stage,
# complex L2 final-only loss). The quantile q_p is taken from the zero-filled |k| of each slice (norm_scope="slice")
# unless stated otherwise.
_AXIAL = {
    **_BASE,
    "prefix": "norm",
    "encoders": ["axial", "axial", "axial"],
    "norm_scope": "slice",
    "final_loss_type": "complex_l2",
    "intermediate_loss_type": "complex_l2",
}

EXPERIMENTS = [
    # Exp: log-quantile normalisation, |k| -> log1p(|k| / q_95) with phase kept (inverse expm1(.) * q_95), per slice.
    {
        **_AXIAL,
        "name": "axial_logq95_slice_l2_final_r4",
        "norm": "log_quantile",
        "norm_quantile": 0.95,
    },
    # Exp: linear fastMRI-magnitude normalisation pinned to the max (p100) of the zero-filled |k| per slice.
    {
        **_AXIAL,
        "name": "axial_p100_slice_l2_final_r4",
        "norm": "fastmri_magnitude",
        "norm_quantile": 1.0,
    },
    # Exp: linear fastMRI-magnitude p95 normalisation, volume-wise (p95 of the whole zero-filled volume from
    # tools/volume_stats.py, cached as <data_dir>/volume_stats.json).
    {
        **_AXIAL,
        "name": "axial_p95_volume_l2_final_r4",
        "norm": "fastmri_magnitude",
        "norm_quantile": 0.95,
        "norm_scope": "volume",
    },
    # Exp: standard per-slice p95 normalisation, but the first 4 and last 4 (noisy edge) slices of every volume are
    # dropped from the TRAINING set (validation still uses every slice).
    {
        **_AXIAL,
        "name": "axial_p95_slice_skip4_4_l2_final_r4",
        "norm": "fastmri_magnitude",
        "norm_quantile": 0.95,
        "skip_starting_slice": 4,
        "skip_ending_slice": 4,
    },
]
