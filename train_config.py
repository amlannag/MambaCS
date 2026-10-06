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

# Image-domain magnitude normalisation: q = p95 of |IFFT(k_zf)|, complex ZF and GT images divided by the same q, FFT'd
# back to k-space for the model. Same axial baseline as Experiments/FASTMRI/Normalisation/Quantile Norm/
# norm_axial_p95_slice_l2_final_r4 (k-space learning, learned lambda, 3x complex axial, no global filter) so the only
# variable is the domain the p95 is taken in. Final-only complex L2 in k-space.
_IMAGE_MAGNITUDE_NORM = {
    **_BASE,
    "prefix": "norm",
    "name": "axial_image_p95_slice_l2_final_r4",
    "attn_type": "complex",
    "encoders": ["axial", "axial", "axial"],
    "norm": "image_magnitude",
    "global_filter": False,
    "kspace_conv": False,
}

# Same model / norm, but the loss is taken in the image domain: IFFT the k-space prediction and GT, then MSE of the
# magnitudes in the normalised units ("image_l2"; the legacy "l2" undoes the normalisation and compares at ~1e-4 scale).
_IMAGE_MAGNITUDE_NORM_IMAGE_L2 = {
    **_IMAGE_MAGNITUDE_NORM,
    "name": "axial_image_p95_slice_image_mag_l2_final_r4",
    "final_loss_type": "image_l2",
    "intermediate_loss_type": "image_l2",
}

EXPERIMENTS = [
    _IMAGE_MAGNITUDE_NORM,
    _IMAGE_MAGNITUDE_NORM_IMAGE_L2,
]
