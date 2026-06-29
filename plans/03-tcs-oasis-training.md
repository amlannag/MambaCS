# Plan 03 — TCS-main Architecture Training on OASIS

## Goal

Add `train_tcs_oasis.py` — a training script that uses **TCS-main's exact model architecture** (Post-LN, `real()` DC, standard `nn.TransformerEncoderLayer`) on the OASIS PNG dataset, with two mask modes to isolate the effect of the undersampling convention. Ship a companion `submit_tcs_oasis.sh` SLURM script that runs both modes sequentially.

---

## Phase 0 — Discovery (DONE)

**Confirmed facts from source reading:**

| Item | Finding |
|---|---|
| TCS-main mask files | `TCS-main/masks/mask_R4.png`, `mask_R6.png`, `mask_R8.png` |
| `MambaCS/masks/` | **EMPTY** — all real mask PNGs are in `TCS-main/masks/` |
| TCS `cascadeNet.__init__` | `(N, encList, encArgs, dcFunc=FFT_DC, lamb=True)` |
| TCS `axVIT.__init__` | `(N, layerNo=2, numCh=1, d_model=None, nhead=8, num_encoder_layers=2, dim_feedforward=None, dropout=0.1, ...)` |
| TCS `axialEncoder.__init__` | `(image_size, numCh=1, d_model=512, nhead=8, num_layers=6, dim_feedforward=None, dropout=0.1, ...)` |
| TCS `FFT_DC` | Non-shifted `fft2`, returns **real part** (`[:,:,:,0:numCh]`), `cy` used in soft-DC branch |
| TCS `FFT_DC` hard-DC branch | Uses raw `y` (real tensor) — **bug present but irrelevant** since we use `lamb=True` |
| `OASISDataset` | `(data_dir, image_size=(256,256))` → yields `[1,H,W]` float32 `[0,1]` |
| `GPUMaskGenerator` | `(accelerations, center_fractions=None)`, `.apply()` → `(kspace_us, mask [1,1,1,W], None)` |
| SLURM params | `--cpus-per-task=8`, `--mem=32G`, `--time=20:00:00`, `gpu_cuda`, `a_ai_collab`, conda env `mambacs` |
| WandB project | `MambaCS-OASIS` (confirmed in existing script) |

---

## Phase 1 — `train_tcs_oasis.py`

### What to build

Single self-contained script at `/Users/amlannag/Desktop/MambaCS/train_tcs_oasis.py`.

**Import strategy** — use `sys.path` to pull TCS-main in without touching it:
```python
import sys, os
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_SCRIPT_DIR, 'TCS-main'))

from DcTNN.tnn import cascadeNet, axVIT       # TCS-main model
from dc.dc   import FFT_DC as TCS_FFT_DC      # TCS-main DC (non-shifted, real() output)
from dc.dc   import fft_2d as tcs_fft_2d
from dc.dc   import ifft_2d as tcs_ifft_2d
```

For GPU-mask mode, import from the existing OASIS script:
```python
from image_domain_testing_brain_mri import (
    OASISDataset, GPUMaskGenerator, simulate_undersampling_png, psnr
)
from DcTNN.dc import FFT_DC as MAMBA_FFT_DC   # MambaCS DC (fftshift, abs() output)
```

### Model construction

Same for both mask modes — TCS-main's `cascadeNet` with 3 axial encoders, 256×256:

```python
N = 256
axArgs = dict(layerNo=1, numCh=1, d_model=None, nhead=8,
              num_encoder_layers=2, dim_feedforward=None)
model = cascadeNet(
    N,
    encList=[axVIT, axVIT, axVIT],
    encArgs=[axArgs, axArgs, axArgs],
    dcFunc=chosen_dc_func,   # TCS_FFT_DC or MAMBA_FFT_DC
    lamb=True,
)
```

### Mode A — `--mask_type tcs` (TCS-main's convention)

```
Preprocessing per batch:
  img [B,1,H,W] float32 [0,1]
    → complex: torch.cat([img, zeros], dim=1) → [B,2,H,W]
    → fft: tcs_fft_2d(img_complex)            → [B,2,H,W] (DC at corner, norm='ortho')
    → mask: fft * mask_2d                      → y [B,2,H,W] (DC reference)
    → ifft: tcs_ifft_2d(y)[:, 0:1, :, :]      → model_input [B,1,H,W] (real part)
    → gt = img

Mask:
  Load once at startup:
    mask_np = np.array(Image.open('TCS-main/masks/mask_R8.png').convert('L'))
    mask_np = np.fft.ifftshift(mask_np) // mask_np.max()   # ACS at corner
    mask = torch.tensor(mask_np, dtype=torch.float32)      # [H,W] on device

Model call:
  recon = model(model_input, y, mask)    # mask [H,W], y [B,2,H,W]
  loss  = F.l1_loss(recon, gt)
```

### Mode B — `--mask_type gpu` (MambaCS convention)

```
Preprocessing per batch:
  img [B,1,H,W] float32 [0,1]
    → simulate_undersampling_png(img, gpu_mask_gen, accel=8)
    → model_input [B,1,H,W], DC_input [B,1,H,W] complex, gt, mask [1,1,1,W]

Model call:
  recon = model(model_input, DC_input, mask)
  loss  = F.l1_loss(recon, gt)
```

`dcFunc = MAMBA_FFT_DC` — the MambaCS version (fftshift, returns `abs()`).

### Optimiser, scheduler, checkpointing

Copy directly from `image_domain_testing_brain_mri.py`:
- Adam lr=1e-4, weight_decay=1e-5
- CosineAnnealingLR(T_max=epochs, eta_min=lr*0.01)
- Save `best_model.pth` (lowest val L1) + `latest.pth`
- Append to `metrics.json` each epoch

### WandB

```python
wandb.init(
    project=args.wandb_project,          # default='MambaCS-OASIS'
    name=f"tcs_arch_8x_{args.mask_type}_mask",
    config=vars(args),
)
```

### CLI interface

```
--train_dir       str   default='/scratch/user/uqanag/OASIS/keras_png_slices_train'
--val_dir         str   default='/scratch/user/uqanag/OASIS/keras_png_slices_validate'
--image_size      int   default=256
--mask_type       str   choices=['tcs','gpu'], required=True
--accel           int   default=8
--epochs          int   default=100
--batch_size      int   default=16
--lr              float default=1e-4
--num_workers     int   default=4
--out_dir         str   default='../Experiments/tcs_oasis_8x'
--wandb_project   str   default='MambaCS-OASIS'
--resume          str   default=None
```

### Verification checklist

- [ ] `python train_tcs_oasis.py --mask_type tcs --epochs 1 --batch_size 2 --train_dir data/oasis_test --val_dir data/oasis_test --num_workers 0` runs to completion without error
- [ ] `python train_tcs_oasis.py --mask_type gpu --epochs 1 --batch_size 2 --train_dir data/oasis_test --val_dir data/oasis_test --num_workers 0` runs to completion without error
- [ ] `grep 'TCS-main' train_tcs_oasis.py` shows path insertion
- [ ] `grep 'norm1\|norm2' train_tcs_oasis.py` returns nothing — model is from TCS-main, not from MambaCS's custom layer
- [ ] Output dir contains `best_model.pth`, `latest.pth`, `metrics.json` after epoch 1

---

## Phase 2 — `submit_tcs_oasis.sh`

Single SLURM script, same header as `submit_oasis.sh`, runs both modes sequentially in one job:

```bash
#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --job-name=tcs_oasis_compare
#SBATCH --time=40:00:00          # doubled for two runs
#SBATCH --qos=gpu
#SBATCH --partition=gpu_cuda
#SBATCH --gres=gpu:1
#SBATCH --account='a_ai_collab'
#SBATCH -o logs/slurm-%j.output
#SBATCH -e logs/slurm-%j.error

export WANDB_API_KEY='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'

module load cuda/11.8.0
module load miniforge/24.11.3-0
source $ROOTMINIFORGE/etc/profile.d/conda.sh
conda activate mambacs

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

unset SLURM_MEM_PER_GPU SLURM_MEM_PER_CPU SLURM_MEM_PER_NODE

echo "=== Run 1: TCS mask convention ==="
srun --cpu-bind=none python train_tcs_oasis.py \
    --train_dir /scratch/user/uqanag/OASIS/keras_png_slices_train \
    --val_dir   /scratch/user/uqanag/OASIS/keras_png_slices_validate \
    --mask_type tcs \
    --accel 8 \
    --epochs 100 \
    --batch_size 16 \
    --out_dir ../Experiments/tcs_oasis_8x_tcs_mask

echo "=== Run 2: GPU column mask convention ==="
srun --cpu-bind=none python train_tcs_oasis.py \
    --train_dir /scratch/user/uqanag/OASIS/keras_png_slices_train \
    --val_dir   /scratch/user/uqanag/OASIS/keras_png_slices_validate \
    --mask_type gpu \
    --accel 8 \
    --epochs 100 \
    --batch_size 16 \
    --out_dir ../Experiments/tcs_oasis_8x_gpu_mask
```

### Verification checklist

- [ ] `sbatch --test-only submit_tcs_oasis.sh` passes on the cluster
- [ ] `grep 'mask_type' submit_tcs_oasis.sh` shows both `tcs` and `gpu` entries
- [ ] `grep 'time' submit_tcs_oasis.sh` shows `40:00:00`

---

## Anti-pattern guards

- Do **not** import from `DcTNN.tnn` in MambaCS — that module has Pre-LN and `abs()` DC. Always import TCS-main's classes via the `sys.path` insertion.
- Do **not** apply `fftshift`/`ifftshift` in Mode A's preprocessing — TCS-main's `fft_2d` and `FFT_DC` assume DC at the corner.
- Do **not** pass `mask [H,W]` to Mode B — `MAMBA_FFT_DC` expects `[1,1,1,W]`.
- Do **not** pass `mask [1,1,1,W]` to Mode A — `TCS_FFT_DC` expects `[H,W]`.
- Do **not** mix DC functions: Mode A must use `TCS_FFT_DC`, Mode B must use `MAMBA_FFT_DC`.
- The `tcs_fft_2d` call produces `[B,2,H,W]` real tensors. The `simulate_undersampling_png` call produces complex tensors. These are incompatible — keep them on separate code paths.
