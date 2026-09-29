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


EXPERIMENTS = [
    # Exp: three PCA-channel stages with axial tokenisation, trained volume-wise: volume-scope PCA (3 volumes per
    # batch) and volume-wise p95 normalisation (tools/volume_stats.py). Each stage splits its input into 3
    # equal-variance PC bins, one axial branch (8 heads, 1 layer) per bin, k-space merge + 1 layer, DC after every
    # stage. Complex L2 on the final output plus unweighted complex L2 on every intermediate stage.
    {
        **_BASE,
        "prefix": "pca",
        "name": "pca_3stage_axial_volnorm_intermediate_l2_r4",
        "encoders": ["pca", "pca", "pca"],
        "pca_tokenizer": "axial",
        "pca_scope": "volume",
        "pca_bins": 3,
        "pca_bin_rule": "equal_variance",
        "pca_detach_basis": True,
        "pca_center": True,
        "pca_layers_per_bin": 1,
        "pca_layers_after_merge": 1,
        "pca_nhead": 8,
        "pca_volumes_per_batch": 3,
        "norm_scope": "volume",
        "loss_mode": "intermediate_unweighted",
        "final_loss_type": "complex_l2",
        "intermediate_loss_type": "complex_l2",
    },
]
