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
    # Shared recipe for the PCA-stage experiments: 3 PCA stages, 3 equal-variance bins, separate branch per bin
    # (8 heads, 1 layer), k-space merge (mean + sum), 1 layer after the merge, DC after every stage, complex L2 final-only.
    # Exp: three PCA-channel stages. Each stage splits its (volume-grouped) input into 3 equal-variance PC bins,
    # runs a separate axial branch (8 heads) per bin, merges in k-space (mean + sum) and runs one more axial
    # layer on the merged k-space; DC after every stage. Volume-scope PCA, 3 volumes per batch.
    {
        **_BASE,
        "prefix": "pca",
        "name": "pca_3stage_axial_volume_fastmri_mag_learned_lambda_l2_final_r4",
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
        "final_loss_type": "complex_l2",
        "intermediate_loss_type": "complex_l2",
    },

    # Exp: same, fixed-APT tokenisation in every branch (apt_layout / apt_embed_dim from _BASE or defaults).
    {
        **_BASE,
        "prefix": "pca",
        "name": "pca_3stage_fixed_apt_volume_fastmri_mag_learned_lambda_l2_final_r4",
        "encoders": ["pca", "pca", "pca"],
        "pca_tokenizer": "fixed_apt",
        "pca_scope": "volume",
        "pca_bins": 3,
        "pca_bin_rule": "equal_variance",
        "pca_detach_basis": True,
        "pca_center": True,
        "pca_layers_per_bin": 1,
        "pca_layers_after_merge": 1,
        "pca_nhead": 8,
        "pca_volumes_per_batch": 3,
        "final_loss_type": "complex_l2",
        "intermediate_loss_type": "complex_l2",
    },

    # Exp: axial PCA stages with BATCH-scope PCA: ordinary shuffled batches, auto batch-size search fits as many
    # slices as the GPU allows (auto_batch_size from _BASE); the PCA is taken over the slices in the batch.
    {
        **_BASE,
        "prefix": "pca",
        "name": "pca_3stage_axial_batchscope_fastmri_mag_learned_lambda_l2_final_r4",
        "encoders": ["pca", "pca", "pca"],
        "pca_tokenizer": "axial",
        "pca_scope": "batch",
        "pca_bins": 3,
        "pca_bin_rule": "equal_variance",
        "pca_detach_basis": True,
        "pca_center": True,
        "pca_layers_per_bin": 1,
        "pca_layers_after_merge": 1,
        "pca_nhead": 8,
        "final_loss_type": "complex_l2",
        "intermediate_loss_type": "complex_l2",
    },
    {
        **_BASE,
        "prefix": "pca",
        "name": "pca_3stage_fixed_apt_batchscope_fastmri_mag_learned_lambda_l2_final_r4",
        "encoders": ["pca", "pca", "pca"],
        "pca_tokenizer": "fixed_apt",
        "pca_scope": "batch",
        "pca_bins": 3,
        "pca_bin_rule": "equal_variance",
        "pca_detach_basis": True,
        "pca_center": True,
        "pca_layers_per_bin": 1,
        "pca_layers_after_merge": 1,
        "pca_nhead": 8,
        "final_loss_type": "complex_l2",
        "intermediate_loss_type": "complex_l2",
    },
]
