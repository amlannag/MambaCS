# MambaCS — Deep Cascade Transformer Neural Network (DcTNN) for MRI Reconstruction

## Project Goal

Reconstruct high-quality MRI images from undersampled k-space measurements using cascaded transformer stages interleaved with physics-based data consistency layers.

---

## Repository Structure

```
MambaCS/
├── config.py              # Config dataclass — all hyperparameters
├── train_config.py        # EXPERIMENTS list of config overrides
├── train.py               # Main training loop (WandB, checkpointing)
├── inference.py           # PDF report generation + evaluation
├── dataset.py             # MRIDataset — loads PNG/TIFF/NPY/NIfTI
├── build_model.py         # Sanity check script (Shepp-Logan phantom)
├── masks/                 # Undersampling masks (R4, R6, R8)
└── DcTNN/
    ├── model.py           # cascadeNet — chains encoders + DC layers
    ├── dc.py              # FFT_DC, KSpace_DC, fft_2d, ifft_2d
    ├── encoders.py        # patchEncoder, axialEncoder, kaleidoscopeEncoder
    ├── vit.py             # patchVIT, axVIT, kaleidoscopeVIT (subclass BaseVIT)
    ├── rope_vit.py        # RoPE utilities + RoPEAttention + RoPETransformerEncoderLayer
    ├── lambda_scheduler.py # LambdaScheduler (constant/linear/cosine schedules)
    └── util.py            # Empty placeholder
```

---

## Architecture: `cascadeNet`

The main model (`DcTNN/model.py`) chains multiple encoder-transformer stages with data consistency between each stage.

### Forward Pass Data Flow

```
GROUND TRUTH IMAGE [B, 1, 320, 320]
        ↓
    FFT → FULL K-SPACE [B, 2, 320, 320] (real + imag channels)
        ↓
    × MASK → UNDERSAMPLED K-SPACE [B, 2, 320, 320]
        ↓
    IFFT → ZERO-FILLED IMAGE [B, 1, 320, 320] (aliased input)
        ↓
    ╔══════════════════════════════════════════════╗
    ║           cascadeNet CASCADE STAGES           ║
    ║  For each stage i:                           ║
    ║    1. im_in = FFT(im) if k-space stage       ║
    ║    2. im_denoise = transformer[i](im_in)     ║
    ║    3. im_out = im_in + im_denoise  (residual)║
    ║    4. im = DC(im_out, y, mask, lambda[i])    ║
    ╚══════════════════════════════════════════════╝
        ↓
RECONSTRUCTED IMAGE [B, 1, 320, 320]
        ↓
    L1 LOSS vs GT → BACKPROP
```

---

## Encoder Types (`DcTNN/encoders.py`)

All encoders share `BaseTokenEncoder` (except `axialEncoder` which is independent).

| Encoder | Tokenization Strategy | Complexity |
|---|---|---|
| `patchEncoder` | Local N×N patches → token per patch | O(N²) attention |
| `kaleidoscopeEncoder` | Globally-spaced pixels → token with global context | O(N²) attention |
| `axialEncoder` | Separate row and column transformer sequences | O(2N) attention |

### ViT Wrappers (`DcTNN/vit.py`)

- `BaseVIT` — shared cascade forward loop
- `patchVIT` → wraps `patchEncoder`
- `axVIT` → wraps `axialEncoder`
- `kaleidoscopeVIT` → wraps `kaleidoscopeEncoder`

---

## Positional Embeddings

All encoder types support three positional embedding modes (controlled via config):

| Mode | Type | Description |
|---|---|---|
| `APE` | Learned absolute | Classic learned positional encodings |
| `Rope-Axial` | Fixed 2D rotary | Separate x and y rotation frequencies |
| `Rope-Mixed` | Learnable per-head rotary | Hybrid: mixed x+y frequencies |

RoPE utilities live in `DcTNN/rope_vit.py`:
- `compute_axial_cis()` — builds 2D axial rotation matrices
- `compute_mixed_cis()` — builds mixed x+y rotation matrices
- `apply_rotary_emb()` — applies rotation to Q and K via complex multiply
- `RoPEAttention` — attention module with RoPE
- `RoPETransformerEncoderLayer` — full transformer layer with RoPE

---

## Data Consistency (`DcTNN/dc.py`)

Physics-based layer applied after each transformer stage.

| Function | Domain | Description |
|---|---|---|
| `FFT_DC` | Image-space | Converts to k-space, enforces consistency, converts back |
| `KSpace_DC` | K-space | Directly enforces consistency in frequency domain |

**Lambda weighting** (soft vs. hard constraint):
- `lamb=None`: hard constraint — `z = (1-mask)*z + mask*y`
- `lamb` learned: soft constraint — `z = (1-mask)*z + mask*(z + λ*y)/(1+λ)`

FFT utilities:
- `fft_2d(x)`: `[B,1,H,W]` → `[B,2,H,W]` (real + imaginary channels)
- `ifft_2d(x)`: inverse

---

## Data Consistency Weight: Lambda

Per-stage lambda can be configured three ways (set in `config.py`):

| Mode | Behavior |
|---|---|
| Learned | `nn.Parameter` per stage, initialized 0.5, trained with model |
| Scheduled | `LambdaScheduler` — decreases over epochs (constant/linear/cosine) |
| None | Hard data consistency constraint, no learnable weight |

`LambdaScheduler` lives in `DcTNN/lambda_scheduler.py`.

---

## Configuration (`config.py` + `train_config.py`)

`Config` is a Python dataclass with all hyperparameters. Key fields:

| Field | Description |
|---|---|
| `encoders` | List of encoder names per cascade stage, e.g. `["axial", "kaleidoscope", "patch"]` |
| `k_space_learning` | Bool list — whether each stage operates in k-space |
| `patch_size` | Patch/kaleidoscope token size |
| `num_encoder_layers` | Transformer layers per stage |
| `nhead_patch`, `nhead_axial` | Attention heads per encoder type |
| `positional_encoding` | `"APE"`, `"Rope-Axial"`, or `"Rope-Mixed"` |
| `lambda_mode` | `"learned"`, `"scheduled"`, or `None` |
| `lambda_schedule` | `"constant"`, `"linear"`, or `"cosine"` |

`train_config.py` contains an `EXPERIMENTS` list — each entry is a dict of overrides applied to the base `Config` at runtime via `setattr()`. Pass experiment index as CLI arg to `train.py`.

---

## Training Pipeline (`train.py`)

1. **Setup** — parse experiment index, apply config overrides, init WandB
2. **Data loading** — load masks for R4/R6/R8 acceleration factors; 90/10 train/val split (deterministic seed)
3. **Model building** — factory pattern via `ENCODER_ARGS` dict maps encoder names → (class, hyperparams)
4. **Optimization** — Adam (lr=1e-4, weight_decay=1e-5) + CosineAnnealingLR (eta_min=lr×0.01)
5. **Train loop** — random mask per batch, simulate undersampling, forward pass, L1 loss, gradient clipping (max_norm=1.0)
6. **Validation** — L1 loss + PSNR, save best checkpoint (lowest val loss)
7. **Post-training** — auto-generate inference PDF reports

---

## Inference / Evaluation (`inference.py`)

Generates multi-page PDF reports including:
- Per-image reconstruction panels (spatial domain + k-space + MSE heatmap)
- Training metric curves
- Summary tables with PSNR/SSIM per acceleration factor

Can batch-process multiple experiments.

---

## Dataset (`dataset.py`)

`MRIDataset`:
- Supports PNG, TIFF, NPY, NIfTI formats
- Resizes images to N×N, normalizes
- Deterministic train/val split via `np.random.RandomState(seed)`

Masks loaded separately from `masks/` directory. Each mask PNG has `np.fft.ifftshift()` applied before use (MRI k-space center convention).

---

## Key Design Decisions

- **Residual denoising**: Transformer predicts a *correction* (`im_denoise`), not a clean image — improves stability and convergence speed.
- **Per-stage domain selection**: Each cascade stage independently chooses image-space or k-space operation via `k_space_learning` boolean list.
- **Per-stage learned λ**: Allows early stages to apply soft DC while later stages can be stricter.
- **Kaleidoscope tokenization**: Globally-spaced pixels per token gives each token immediate global context — contrasts with local patch tokens.
- **Axial attention**: Row and column sequences processed separately, reducing attention complexity from O(N²) to O(2N).
- **Complex FFT via real tensors**: Explicit [B,2,H,W] real/imag channel split instead of PyTorch complex tensor type — allows more explicit control.
- **No BatchNorm/InstanceNorm in transformers**: Only LayerNorm (within `TransformerEncoder`) — keeps design minimal.

---

## Sanity Check (`build_model.py`)

Generates a Shepp-Logan phantom, applies undersampling, runs a full cascade forward pass, visualizes output, and reports trainable parameter count. Use this to validate architecture changes before running full training.
