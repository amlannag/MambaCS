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

# k-space learning with the loss taken on the COMPLEX image: IFFT the k-space prediction and compare Re/Im against the
# normalised GT complex image ("complex_image_l2"). Phase-sensitive, unlike "image_l2" which only compares magnitudes.
# (With the ortho FFT this equals complex_l2 over all of k-space by Parseval; kept as an explicit image-domain control.)
_IMAGE_MAGNITUDE_NORM_COMPLEX_IMAGE_L2 = {
    **_IMAGE_MAGNITUDE_NORM,
    "name": "axial_image_p95_slice_complex_image_l2_final_r4",
    "final_loss_type": "complex_image_l2",
    "intermediate_loss_type": "complex_image_l2",
}

# k-space learning with a DcCNN-style residual COMPLEX CNN in the image domain at the start of every encoder stage:
# IFFT(k) -> conv(1->32)+CReLU -> conv(32->32)+CReLU -> conv(32->1) (zero-init, identity at start) -> residual -> FFT.
# Loss stays the baseline complex L2 in k-space, so the only variable vs _IMAGE_MAGNITUDE_NORM is the stem.
_IMAGE_MAGNITUDE_NORM_IMAGE_CONV = {
    **_IMAGE_MAGNITUDE_NORM,
    "name": "axial_image_p95_slice_imgconv32x3_l2_final_r4",
    "image_conv": True,
    "image_conv_channels": 32,
    "image_conv_layers": 3,
    "image_conv_kernel": 3,
}

# Both together: image-domain CNN stem per stage AND the complex-image L2 loss.
_IMAGE_MAGNITUDE_NORM_IMAGE_CONV_COMPLEX_IMAGE_L2 = {
    **_IMAGE_MAGNITUDE_NORM_IMAGE_CONV,
    "name": "axial_image_p95_slice_imgconv32x3_complex_image_l2_final_r4",
    "final_loss_type": "complex_image_l2",
    "intermediate_loss_type": "complex_image_l2",
}

EXPERIMENTS = [
    _IMAGE_MAGNITUDE_NORM_COMPLEX_IMAGE_L2,
    _IMAGE_MAGNITUDE_NORM_IMAGE_CONV,
    _IMAGE_MAGNITUDE_NORM_IMAGE_CONV_COMPLEX_IMAGE_L2,
]
