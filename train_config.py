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
    # Ring-phase + magnitude loss: SNR-weighted 1-cos(dphi) averaged per radial ring, rings averaged, plus (|gt|-|pred|)^2.
    # Volume-wise normalisation (p95 of the whole zero-filled volume) and volume noise sigma (tools/volume_stats.py).
    {
        **_BASE,
        "prefix": "ring_phase",
        "name": "ring_phase_fixed_magL2_volnorm_final_r4",
        "encoders": ["axial", "axial", "axial"],
        "norm_scope": "volume",
        "final_loss_type": "ring_phase_mag",
        "intermediate_loss_type": "ring_phase_mag",
        "ring_phase_weighting": "fixed",
        "ring_phase_weight": 1.0,
        "ring_phase_edges": [0.05, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0, 1.42],
    },
    # Same with the learnable per-ring weighting  (1/K) sum_k ( exp(-s_k) P_k + s_k ),  s_k init 0.
    {
        **_BASE,
        "prefix": "ring_phase",
        "name": "ring_phase_learnable_magL2_volnorm_final_r4",
        "encoders": ["axial", "axial", "axial"],
        "norm_scope": "volume",
        "final_loss_type": "ring_phase_mag",
        "intermediate_loss_type": "ring_phase_mag",
        "ring_phase_weighting": "learnable",
        "ring_phase_edges": [0.05, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0, 1.42],
    },
]
