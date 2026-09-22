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
    {
        **_BASE,
        "prefix": "perpendicular_loss",
        "name": "perpendicular_gtscaled_phaseL2_magL2_final_r4",
        "encoders": ["axial", "axial", "axial"],
        "final_loss_type": "perpendicular_loss",
        "intermediate_loss_type": "perpendicular_loss",
        "perpendicular_magnitude_norm": "l2",
        "perpendicular_phase_norm": "l2",
        "perpendicular_phase_scale": "gt",
        "perpendicular_r_boundary": None,
        "perpendicular_branch_multiplier": 1.0,
    },
]
