# Transformer Compressed Sensing via Global Image Tokens

PyTorch implementation of the **Deep Cascade of Transformer Neural Networks (DcTNN)** for accelerated MRI reconstruction. The model cascades axial, kaleidoscope, and patch transformer encoders with learned data consistency to reconstruct images from undersampled k-space measurements.

---

## Repository Structure

```
MambaCS/
├── DcTNN/
│   ├── encoders.py          # Encoder modules: imageEncoder, axialEncoder, kaleidoscopeEncoder
│   ├── vit.py               # ViT wrappers: patchVIT, axVIT, kaleidoscopeVIT
│   ├── model.py             # cascadeNet — chains encoders with data consistency
│   ├── dc.py                # Data consistency: FFT_DC, KSpace_DC
│   ├── rope_vit.py          # RoPE utilities: axial, mixed 2D frequencies
│   └── lambda_scheduler.py  # Cosine/linear/constant lambda schedules
├── config.py                # Config dataclass with all hyperparameters
├── train_config.py          # Experiment override definitions
├── train.py                 # Training pipeline (WandB logging, PSNR, checkpointing)
├── inference.py             # Inference and evaluation
├── dataset.py               # MRIDataset loader
├── build_model.py           # Quick model build and sanity-check script
└── masks/                   # Undersampling masks (mask_R4.png, mask_R6.png, mask_R8.png)
```

---

## Environment Dependencies

- Python 3.8+
- PyTorch 1.9+
- Einops 0.3+
- Pillow 8.0+
- WandB (for training logging)
- Phantominator (for `build_model.py` sanity check)

---

## Encoder Types

Three encoder types can be composed freely in the cascade:

| Config string    | Class             | Description                                                           |
|------------------|-------------------|-----------------------------------------------------------------------|
| `"axial"`        | `axVIT`           | Row and column transformers — captures global structure               |
| `"kaleidoscope"` | `kaleidoscopeVIT` | Each token samples globally spaced pixels at a fixed sub-patch offset |
| `"patch"`        | `patchVIT`        | Standard local patch tokens — captures fine texture                   |

Positional embedding options (`pos_emb_type`):

- `"APE"` — learned absolute positional embedding
- `"Rope-Axial"` — fixed 2D axial rotary embeddings
- `"Rope-Mixed"` — learnable per-head rotary embeddings with random initial rotations

---

## Training

All hyperparameters are defined in `config.py`. Experiments are defined as override dicts in `train_config.py`.

```bash
python train.py --exp_idx 0
```

Key config fields:

| Field                  | Default  | Description                                                              |
|------------------------|----------|--------------------------------------------------------------------------|
| `encoders`             | —        | Ordered list of encoder stages, e.g. `["axial", "kaleidoscope", "patch"]` |
| `patch_size`           | `16`     | Patch/token size                                                         |
| `layer_no`             | `1`      | Cascaded denoising blocks within each ViT                                |
| `num_encoder_layers`   | `2`      | Transformer layers per encoder block                                     |
| `pos_emb_type`         | `"APE"`  | Positional embedding type                                                |
| `acceleration_factors` | `[8]`    | Undersampling rates to train on                                          |
| `k_space_learning`     | `False`  | Bool or per-stage list — whether each stage operates in k-space          |
| `lambda_schedule`      | `"none"` | `"none"` (learned) or `"cosine"` / `"linear"` / `"constant"`            |

Example `encoders` configurations:

```python
["axial", "kaleidoscope", "patch"]          # Original DcTNN (3 stages)
["axial", "patch"]                           # 2-stage, no kaleidoscope
["patch", "patch", "patch"]                  # Patch-only ablation
["axial", "kaleidoscope", "patch", "patch"]  # 4-stage deeper model
```

---

## Quick Start / Sanity Check

`build_model.py` builds the default cascade and runs a forward pass on a Shepp-Logan phantom:

```bash
python build_model.py
```

---

## Programmatic Usage

```python
from DcTNN.model import cascadeNet, axVIT, kaleidoscopeVIT, patchVIT

N = 320
encList = [axVIT, kaleidoscopeVIT, patchVIT]
encArgs = [
    {"layerNo": 1, "numCh": 1, "nhead": 8, "num_encoder_layers": 2, "dim_feedforward": None},
    {"patch_size": 16, "layerNo": 1, "numCh": 1, "nhead": 8, "num_encoder_layers": 2, "dim_feedforward": None},
    {"patch_size": 16, "layerNo": 1, "numCh": 1, "nhead": 8, "num_encoder_layers": 2, "dim_feedforward": None},
]

model = cascadeNet(N, encList, encArgs, lamb=True)
```

---

## Citation and Acknowledgement

Paper available on [IEEE Xplore](https://doi.org/10.1109/ICIP46576.2022.9897630):
```
M. B. Lorenzana, C. Engstrom and S. S. Chandra, "Transformer Compressed Sensing Via Global Image Tokens,"
2022 IEEE International Conference on Image Processing (ICIP), 2022, pp. 3011-3015.
```

Paper also available on [arXiv](https://arxiv.org/abs/2203.12861):
```
M. Bran Lorenzana, C. Engstrom, and S. S. Chandra, "Transformer Compressed Sensing via Global Image Tokens."
arXiv, Mar. 2022.
```

Kaleidoscope transform introduced by [White et al.](https://doi.org/10.1109/LSP.2021.3116510):
```
J. M. White, S. Crozier and S. S. Chandra, "Bespoke Fractal Sampling Patterns for Discrete Fourier Space
via the Kaleidoscope Transform," IEEE Signal Processing Letters, vol. 28, pp. 2053-2057, 2021.
```
