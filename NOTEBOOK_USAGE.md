# FastMRI Phase-Aware Inference Notebook - Multi-Experiment Support

## Overview
The `fastmri_phase_aware_inference.ipynb` notebook has been updated to support running and comparing multiple experiments with custom presentation names.

## How to Use

### 1. Define Your Experiments
Edit the `EXPERIMENTS` dictionary in the config cell:

```python
EXPERIMENTS = {
    "exp1": {
        "path": "fastMRI_phase_aware_kspace_none_perpendicular_loss_weighted_m1_to_m4_linear_4x",
        "name": "Phase-Aware (4x)"
    },
    "exp2": {
        "path": "fastMRI_other_experiment_name",
        "name": "Baseline Model"
    },
    "exp3": {
        "path": "fastMRI_another_experiment",
        "name": "Cross-Axial Variant"
    },
}
```

**Key points:**
- `exp_id`: Internal identifier (can be anything, not displayed)
- `path`: Directory name under `Experiments/` folder
- `name`: Display name used in all plots and outputs

### 2. Run the Notebook
Execute cells in order:
1. Setup & Imports
2. Config (update EXPERIMENTS dict)
3. Helpers (loads metric functions)
4. Run All (runs all experiments on all slices)
5. Comparison Plots (shows metrics side-by-side)
6. Visualization cells (shows reconstructions side-by-side)

### 3. View Results

#### Experiment Summary
After running "Run All", you'll see:
- Per-slice metrics for each experiment
- Mean metrics summary showing all experiments

#### Comparison Bar Charts
Shows PSNR and SSIM metrics across all experiments with presentation names.

#### Side-by-Side Visualizations
- Image-domain magnitude (GT | Zero-fill | Recon) for all slices
- Phase information
- K-space magnitude and phase

All plots use presentation names (from the `"name"` field) instead of directory names.

## Example: Adding an Experiment

To add a third experiment, simply edit the EXPERIMENTS dictionary:

```python
EXPERIMENTS = {
    "exp1": {
        "path": "fastMRI_phase_aware_kspace_none_perpendicular_loss_weighted_m1_to_m4_linear_4x",
        "name": "Phase-Aware (4x)"
    },
    "exp2": {
        "path": "fastMRI_k_space_learned_lambda",
        "name": "Learned λ (4x)"
    },
}
```

Then run the notebook again. All experiments will be evaluated and compared automatically.

## Key Changes Made

1. **Config cell**: Loads all experiments in a loop instead of a single hardcoded path
2. **Helpers**: `run_slice()` now takes `exp_id` parameter to use correct model/config
3. **Run All cell**: Processes all experiments, stores results in `all_results` dict
4. **New Comparison cell**: Bar plots of mean metrics with presentation names
5. **Plot cells**: Updated to iterate through experiments and display side-by-side

## Output Structure

Results are stored as:
```python
all_results = {
    "exp1": {
        "exp_name": "Phase-Aware (4x)",
        "results": [slice1_data, slice2_data, ...],
        "mean_metrics": {
            "zf_psnr": value,
            "zf_ssim": value,
            "recon_psnr": value,
            "recon_ssim": value,
        }
    },
    ...
}
```

Each slice's data bundle contains:
- Image-domain data (magnitude, phase, complex)
- K-space data (magnitude, phase, complex)
- Mask and reconstruction metrics
