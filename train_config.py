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
    # Exp 3: plain reverse-Huber (Berhu) on the complex k-space error: |e| for |e| <= 1, (e^2 + 1)/2 above.
    # Same 3x axial / learned lambda / fastmri_magnitude / final-only recipe as the other loss experiments.
    {
        **_BASE,
        "prefix": "berhu_loss",
        "name": "berhu_d1_loss_final_r4",
        "encoders": ["axial", "axial", "axial"],
        "final_loss_type": "complex_berhu",
        "intermediate_loss_type": "complex_berhu",
        "berhu_delta": 1.0,
    },
]
