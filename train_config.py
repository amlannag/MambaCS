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
    # All three: 3x axial cascade (no cross-attention, no FNet), learned lambda DC, fastmri_magnitude, final-only loss.

    # Exp 0: data-driven ring-weighted complex L2. Radial rings de-emphasise the sampled centre and boost
    # r >= 60% of the radius (x3); extra horizontal factor on |kx| >= 60% (x2) since the Cartesian mask varies
    # along kx. On the L2 baseline's error: loss share r>=0.6 63% -> 86%, |kx|>=0.6 34% -> 54% (12x weight range).
    {
        **_BASE,
        "prefix": "ring_weighted_l2",
        "name": "dctnn_axial_3axial_fastmri_mag_learned_lambda_ringw_l2_r60_kx60_final_r4",
        "encoders": ["axial", "axial", "axial"],
        "final_loss_type": "freq_weighted_complex_l2",
        "intermediate_loss_type": "freq_weighted_complex_l2",
        "freq_weight_form": "rings",
        "freq_weight_ring_edges": [0.0, 0.1, 0.2, 0.3, 0.6, 1.0, 1.42],       # px: 0, 16, 32, 48, 96, 160, 227
        "freq_weight_ring_weights": [0.5, 0.5, 1.0, 1.5, 3.0, 3.0],
        "freq_weight_kx_ring_edges": [0.0, 0.1, 0.3, 0.6, 1.0],                # px: 0, 16, 48, 96, 160
        "freq_weight_kx_ring_weights": [1.0, 1.0, 1.5, 2.0],
    },

    # Exp 1: complex L2 + LORAKS C-matrix low-rank prior with weight 1. normalize="mean" is the un-normalised
    # tail energy whose gradient matches the classical LORAKS MM update (the "ratio" form adds a term that
    # rewards inflating the top singular values). R=3 -> Nr=29, rank None -> 14.
    {
        **_BASE,
        "prefix": "loraks_l2",
        "name": "dctnn_axial_3axial_fastmri_mag_learned_lambda_l2_loraks_w1_mean_final_r4",
        "encoders": ["axial", "axial", "axial"],
        "final_loss_type": "complex_l2_loraks",
        "intermediate_loss_type": "complex_l2_loraks",
        "loraks_weight": 1.0,
        "loraks_radius": 3,
        "loraks_rank": None,
        "loraks_normalize": "mean",
    },

    # Exp 2: perpendicular loss with L2 on the magnitude component (phase term unchanged, no magnitude weighting).
    {
        **_BASE,
        "prefix": "perpendicular_loss",
        "name": "dctnn_axial_3axial_fastmri_mag_learned_lambda_perpendicular_magL2_final_r4",
        "encoders": ["axial", "axial", "axial"],
        "final_loss_type": "perpendicular_loss",
        "intermediate_loss_type": "perpendicular_loss",
        "perpendicular_magnitude_norm": "l2",
        "perpendicular_mag_weighting": False,
    },
]
