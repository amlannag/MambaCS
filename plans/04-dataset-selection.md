# Plan 04 — Dataset Selection

**Goal:** Add a `dataset` field to `Config` so each experiment can declare `"fastmri"` or `"oasis"`. The chosen dataset routes W&B logging to a matching project name and instantiates the correct `Dataset` class. Everything else in the pipeline (mask generation, undersampling simulation, normalisation, model, loss) is untouched.

**Files changed (4 total):**
- `config.py` — add `dataset` field
- `dataset.py` — add `OASISDataset` class
- `train.py` — W&B project routing, dataset factory, fix `.h5_files` reference
- `train_config.py` — add `"dataset"` key to existing experiment and add OASIS example

---

## Phase 0 — Document Discovery Summary

This section records what was found by reading each file before any edits were made. An agent executing this plan should trust these findings and skip re-reading unless something looks wrong.

### `config.py` (102 lines)

- `Config` is a plain `@dataclass` with no `__post_init__`.
- The **Data section** starts at line 33 (`# Data`) and currently contains:
  - Line 36: `data_dir: str = "/scratch/user/uqanag/fastmri/singlecoil_train"`
  - Line 37: `val_data_dir: Optional[str] = "/scratch/user/uqanag/fastmri/singlecoil_val"`
  - Line 39: `kspace_key: str = "kspace"`
  - Line 40: `image_size: Tuple[int, int] = (320, 320)`
  - Line 50: `max_val_files: Optional[int] = 15`
- There is no `dataset` field anywhere in the file. One field must be added.

### `dataset.py` (75 lines)

- Imports: `os`, `h5py`, `torch`, `torch.utils.data.Dataset`, `torch.utils.data.get_worker_info`.
- `center_crop_kspace` helper lives at lines 8–14.
- `H5MRIDataset` occupies lines 17–74 and exposes a `.h5_files` attribute (list of absolute file paths, set at line 44).
- No PIL, no image-file scanning — all of that is new.
- File ends at line 75 with a blank line after `H5MRIDataset.__del__`.

### `train.py` (307 lines)

- Line 17: `from dataset import H5MRIDataset` — the only dataset import.
- Lines 164–168: `wandb.init(project="MambaCS", ...)` — hardcoded project name, 5 lines.
- Lines 178–199 (datasets block):
  - Line 179: `train_data_dir, val_data_dir = resolve_data_dirs(cfg)`
  - Lines 180–181: `train_ds = H5MRIDataset(train_data_dir, image_size=cfg.image_size, kspace_key=cfg.kspace_key)`
  - Lines 182–184: `val_ds = H5MRIDataset(val_data_dir, image_size=cfg.image_size, kspace_key=cfg.kspace_key, max_files=cfg.max_val_files)`
  - Line 199: `print(f"Val files   : {len(val_ds.h5_files)} capped")` — `.h5_files` is hardcoded; breaks for `OASISDataset`.

### `train_config.py` (16 lines)

- `EXPERIMENTS` is a list with one dict (lines 6–16).
- The single experiment has keys: `prefix`, `name`, `encoders`, `learning`, `norm`, `pos_emb_type`, `attn_type`, `acceleration_factors`.
- No `dataset` key. The `build_cfg` function in `train.py` (line 33) raises `ValueError` for any key not present on `Config`, so `"dataset"` must be on `Config` before the config file is updated.

---

## Phase 1 — `config.py`: Add `dataset` field

### What to change

Add one field to the Data section of `Config`, immediately before the `data_dir` line (line 36). Also update the `data_dir` comment so it is dataset-agnostic.

### Exact edit

**File:** `/Users/amlannag/Desktop/MambaCS/config.py`

Replace lines 34–36 (the Data comment and `data_dir` line):

```
    # ---------------------------------------------------------------------------
    # Data
    # ---------------------------------------------------------------------------

    # Folder of training .h5 MRI files (fastMRI format, one file per scan)
    data_dir: str = "/scratch/user/uqanag/fastmri/singlecoil_train"
```

With:

```python
    # ---------------------------------------------------------------------------
    # Data
    # ---------------------------------------------------------------------------

    # Which dataset to use. Determines the Dataset class and W&B project name.
    # "fastmri" — loads .h5 files via H5MRIDataset (default)
    # "oasis"   — loads PNG brain slices via OASISDataset
    dataset: str = "fastmri"

    # Root folder of training data files (applies to whichever dataset is active)
    data_dir: str = "/scratch/user/uqanag/fastmri/singlecoil_train"
```

### Full old_string / new_string for the Edit tool

old_string:
```
    # ---------------------------------------------------------------------------
    # Data
    # ---------------------------------------------------------------------------

    # Folder of training .h5 MRI files (fastMRI format, one file per scan)
    data_dir: str = "/scratch/user/uqanag/fastmri/singlecoil_train"
```

new_string:
```
    # ---------------------------------------------------------------------------
    # Data
    # ---------------------------------------------------------------------------

    # Which dataset to use. Determines the Dataset class and W&B project name.
    # "fastmri" — loads .h5 files via H5MRIDataset (default)
    # "oasis"   — loads PNG brain slices via OASISDataset
    dataset: str = "fastmri"

    # Root folder of training data files (applies to whichever dataset is active)
    data_dir: str = "/scratch/user/uqanag/fastmri/singlecoil_train"
```

### Verification

```bash
python -c "from config import Config; c = Config(); assert c.dataset == 'fastmri', c.dataset; print('PASS: dataset defaults to fastmri')"
```

Expected output: `PASS: dataset defaults to fastmri`

---

## Phase 2 — `dataset.py`: Add `OASISDataset`

### What to change

Append `OASISDataset` after the existing `H5MRIDataset` class (after line 75). Also add `PIL` to the imports at the top of the file.

### Step 2a — Add PIL import

**File:** `/Users/amlannag/Desktop/MambaCS/dataset.py`

old_string:
```
import os
import h5py
import torch
from torch.utils.data import Dataset
from torch.utils.data import get_worker_info
```

new_string:
```
import os
import h5py
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import get_worker_info
```

### Step 2b — Append `OASISDataset` class

Append the following block to the very end of `/Users/amlannag/Desktop/MambaCS/dataset.py` (after the blank line that closes `H5MRIDataset`):

```python

class OASISDataset(Dataset):
    """
    Loads grayscale brain MRI slices from a directory of PNG/JPEG images
    (e.g. the OASIS keras_png_slices dataset) and returns fake k-space data
    compatible with the rest of the MambaCS pipeline.

    Each image is:
      1. Opened in grayscale mode with PIL.
      2. Resized to `image_size` using LANCZOS resampling.
      3. Normalised to [0, 1] as float32.
      4. Converted to a torch tensor of shape [H, W].
      5. Transformed to k-space via fftshift(fft2(ifftshift(image), norm='ortho')) — matches fft_2d convention.
      6. Unsqueezed to [1, H, W] complex64.

    This output shape and dtype match H5MRIDataset.__getitem__, so
    FastMRIMaskGenerator, simulate_undersampling, and normalizer.py require
    no changes.

    Args:
        data_dir  (str):              Directory containing image files.
        image_size (tuple[int,int]):  Output spatial size (H, W). Default (256, 256).
        max_files (int | None):       Cap the number of files loaded (sorted order).
    """

    def __init__(self, data_dir, image_size=(256, 256), max_files=None):
        self.image_size = image_size

        _EXTS = {'.png', '.jpg', '.jpeg'}
        image_files = sorted(
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if os.path.splitext(f)[1].lower() in _EXTS
        )
        if not image_files:
            raise ValueError(f"No PNG/JPEG files found in {data_dir}")
        if max_files is not None:
            image_files = image_files[:max_files]

        # Exposed as .image_files so train.py can print the count (mirrors H5MRIDataset.h5_files)
        self.image_files = image_files

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        fpath = self.image_files[idx]
        img = Image.open(fpath).convert('L')                              # grayscale PIL image
        img = img.resize((self.image_size[1], self.image_size[0]),        # PIL uses (W, H)
                         Image.LANCZOS)
        img_t = torch.tensor(
            __import__('numpy').array(img, dtype='float32') / 255.0       # [H, W] in [0, 1]
        )
        # IMPORTANT: must match fft_2d/ifft_2d convention in DcTNN/dc.py:
        #   fft_2d(x) = fftshift(fft2(ifftshift(x), norm='ortho'))
        # Without ifftshift before fft2 and norm='ortho', ifft_2d will return
        # a quadrant-swapped image scaled by sqrt(N) instead of the original.
        kspace = torch.fft.fftshift(
            torch.fft.fft2(torch.fft.ifftshift(img_t), norm='ortho')
        )                                                                  # [H, W] complex64
        return kspace.unsqueeze(0).to(torch.complex64)                    # [1, H, W]
```

**Note on the numpy import:** The inline `__import__('numpy')` avoids adding a top-level `import numpy` that isn't otherwise needed by this file. If the project already imports numpy globally it is cleaner to add `import numpy as np` to the imports block and use `np.array(img, dtype='float32')` instead.

**Note on FFT convention:** `fft_2d` and `ifft_2d` in `DcTNN/dc.py` both apply `ifftshift` before the transform and `fftshift` after. The OASIS fake k-space must use the same convention so that `ifft_2d(oasis_kspace) = original_image`. Using `fftshift(fft2(img))` without `ifftshift` and `norm='ortho'` would make `ifft_2d` return `fftshift(img) * sqrt(N)` — spatially garbled and wrongly scaled.

### Verification

```bash
cd /Users/amlannag/Desktop/MambaCS
python -c "
from dataset import OASISDataset
print('PASS: OASISDataset imported successfully')
"
```

Expected output: `PASS: OASISDataset imported successfully`

For a more thorough smoke-test (requires a directory of PNGs):

```bash
python -c "
from dataset import OASISDataset
import torch
ds = OASISDataset('/path/to/png/dir', image_size=(256, 256), max_files=2)
sample = ds[0]
assert sample.shape == torch.Size([1, 256, 256]), sample.shape
assert sample.dtype == torch.complex64, sample.dtype
print(f'PASS: shape={sample.shape}, dtype={sample.dtype}')
"
```

---

## Phase 3 — `train.py`: W&B routing + dataset factory + `.h5_files` fix

Three independent sub-changes, presented in the order they appear in the file.

### Step 3a — Update dataset import (line 17)

**File:** `/Users/amlannag/Desktop/MambaCS/train.py`

old_string:
```
from dataset import H5MRIDataset
```

new_string:
```
from dataset import H5MRIDataset, OASISDataset
```

### Step 3b — W&B project routing (lines 164–168)

old_string (exact 5 lines):
```
    wandb.init(
        project="MambaCS",
        name=f"{cfg.prefix}_{cfg.name}",
        config=config_to_dict(cfg),
    )
```

new_string:
```
    _WANDB_PROJECT = {"fastmri": "fastMRI", "oasis": "OASIS"}
    wandb.init(
        project=_WANDB_PROJECT.get(cfg.dataset, "MambaCS"),
        name=f"{cfg.prefix}_{cfg.name}",
        config=config_to_dict(cfg),
    )
```

### Step 3c — Dataset factory and `.h5_files` fix (lines 179–199)

Replace the entire datasets block from the `resolve_data_dirs` call through the `print` of the capped val-files count.

old_string (lines 178–199):
```
    # ---- Datasets ----
    train_data_dir, val_data_dir = resolve_data_dirs(cfg)
    train_ds = H5MRIDataset(train_data_dir, image_size=cfg.image_size,
                            kspace_key=cfg.kspace_key)
    val_ds   = H5MRIDataset(val_data_dir, image_size=cfg.image_size,
                            kspace_key=cfg.kspace_key,
                            max_files=cfg.max_val_files)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              shuffle=True,  num_workers=cfg.num_workers,
                              pin_memory=True,
                              persistent_workers=cfg.num_workers > 0)

    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size,
                            shuffle=False, num_workers=cfg.num_workers,
                            pin_memory=True,
                            persistent_workers=cfg.num_workers > 0)

    print(f"Train dir   : {train_data_dir}")
    print(f"Val dir     : {val_data_dir}")
    if cfg.max_val_files is not None:
        print(f"Val files   : {len(val_ds.h5_files)} capped")
    print(f"Train / Val : {len(train_ds)} / {len(val_ds)} samples")
```

new_string:
```
    # ---- Datasets ----
    train_data_dir, val_data_dir = resolve_data_dirs(cfg)

    def _make_dataset(cfg, data_dir, max_files=None):
        """Return the correct Dataset subclass based on cfg.dataset."""
        if cfg.dataset == "oasis":
            return OASISDataset(data_dir, image_size=cfg.image_size, max_files=max_files)
        return H5MRIDataset(data_dir, image_size=cfg.image_size,
                            kspace_key=cfg.kspace_key, max_files=max_files)

    train_ds = _make_dataset(cfg, train_data_dir)
    val_ds   = _make_dataset(cfg, val_data_dir, max_files=cfg.max_val_files)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              shuffle=True,  num_workers=cfg.num_workers,
                              pin_memory=True,
                              persistent_workers=cfg.num_workers > 0)

    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size,
                            shuffle=False, num_workers=cfg.num_workers,
                            pin_memory=True,
                            persistent_workers=cfg.num_workers > 0)

    print(f"Train dir   : {train_data_dir}")
    print(f"Val dir     : {val_data_dir}")
    file_list = getattr(val_ds, 'h5_files', None) or getattr(val_ds, 'image_files', None)
    if cfg.max_val_files is not None and file_list is not None:
        print(f"Val files   : {len(file_list)} capped")
    print(f"Train / Val : {len(train_ds)} / {len(val_ds)} samples")
```

### Verification

```bash
# Confirm the hardcoded project name is gone
grep -n 'project="MambaCS"' /Users/amlannag/Desktop/MambaCS/train.py \
  && echo "FAIL: hardcoded project still present" \
  || echo "PASS: hardcoded project removed"

# Confirm OASISDataset is imported
grep -n 'OASISDataset' /Users/amlannag/Desktop/MambaCS/train.py \
  && echo "PASS: OASISDataset referenced in train.py" \
  || echo "FAIL: OASISDataset not found in train.py"

# Confirm .h5_files direct attribute access is gone from the print line
grep -n 'val_ds\.h5_files' /Users/amlannag/Desktop/MambaCS/train.py \
  && echo "FAIL: direct .h5_files access still present" \
  || echo "PASS: direct .h5_files access removed"
```

---

## Phase 4 — `train_config.py`: Add `dataset` keys

### What to change

**File:** `/Users/amlannag/Desktop/MambaCS/train_config.py`

Replace the entire `EXPERIMENTS` list:

old_string:
```
EXPERIMENTS = [
    {
        "prefix": "L1",
        "name": "image_axial_APE_4x",
        "encoders": ["axial", "axial", "axial"],
        "learning": "image",
        "norm": None,
        "pos_emb_type": "APE",
        "attn_type": "standard",
        "acceleration_factors": [4],
    },
]
```

new_string:
```
EXPERIMENTS = [
    # --- fastMRI baseline (experiment index 0) ---
    {
        "prefix": "L1",
        "name": "image_axial_APE_4x",
        "dataset": "fastmri",
        "encoders": ["axial", "axial", "axial"],
        "learning": "image",
        "norm": None,
        "pos_emb_type": "APE",
        "attn_type": "standard",
        "acceleration_factors": [4],
    },
    # --- OASIS brain MRI (experiment index 1) ---
    {
        "prefix": "L1",
        "name": "oasis_axial_APE_4x",
        "dataset": "oasis",
        "data_dir": "/scratch/user/uqanag/OASIS/keras_png_slices_train",
        "val_data_dir": "/scratch/user/uqanag/OASIS/keras_png_slices_validate",
        "image_size": (256, 256),
        "encoders": ["axial", "axial", "axial"],
        "learning": "image",
        "norm": None,
        "pos_emb_type": "APE",
        "attn_type": "standard",
        "acceleration_factors": [4],
    },
]
```

**Key points:**
- The fastMRI entry is unchanged except for the added `"dataset": "fastmri"` key. It still uses the `data_dir` and `val_data_dir` from `Config` defaults.
- The OASIS entry explicitly sets `data_dir`, `val_data_dir`, and `image_size` because the OASIS paths and image resolution differ from the fastMRI defaults.
- `build_cfg` in `train.py` (line 33) now accepts `"dataset"` because `Config` has the field after Phase 1.

### Verification

```bash
python -c "
from train_config import EXPERIMENTS
assert all('dataset' in e for e in EXPERIMENTS), 'Missing dataset key in some experiment'
assert EXPERIMENTS[0]['dataset'] == 'fastmri'
assert EXPERIMENTS[1]['dataset'] == 'oasis'
print(f'PASS: {len(EXPERIMENTS)} experiments, all have dataset key')
"
```

Expected output: `PASS: 2 experiments, all have dataset key`

---

## Phase 5 — Full Verification Checklist

Run all checks from the project root (`/Users/amlannag/Desktop/MambaCS`):

### Check 1 — OASISDataset imports without error

```bash
python -c "from dataset import OASISDataset; print('PASS: OASISDataset imported')"
```

### Check 2 — Config default is `"fastmri"`

```bash
python -c "
from config import Config
c = Config()
assert c.dataset == 'fastmri', f'Expected fastmri, got {c.dataset!r}'
print('PASS: Config().dataset == fastmri')
"
```

### Check 3 — All experiments carry a `dataset` key

```bash
python -c "
from train_config import EXPERIMENTS
assert all('dataset' in e for e in EXPERIMENTS), 'Some experiment is missing dataset key'
print(f'PASS: all {len(EXPERIMENTS)} experiments have dataset key')
"
```

### Check 4 — W&B no longer hardcodes `project="MambaCS"`

```bash
grep -n 'project="MambaCS"' /Users/amlannag/Desktop/MambaCS/train.py \
  && echo "FAIL" || echo "PASS: no hardcoded project name"
```

### Check 5 — `build_cfg` accepts the OASIS experiment without ValueError

```bash
python -c "
import sys; sys.argv = ['train.py', '--exp_idx', '1']
# We just want to test that build_cfg doesn't throw; importing triggers it at module level.
# Use direct call instead:
from config import Config
from train_config import EXPERIMENTS
cfg = Config()
for key, val in EXPERIMENTS[1].items():
    assert hasattr(cfg, key), f'Unknown key: {key}'
    setattr(cfg, key, val)
assert cfg.dataset == 'oasis'
assert cfg.image_size == (256, 256)
print('PASS: OASIS experiment config resolves cleanly')
"
```

### Check 6 — H5MRIDataset is still importable (no regression)

```bash
python -c "from dataset import H5MRIDataset; print('PASS: H5MRIDataset still importable')"
```

---

## Execution Order

The phases must be executed in this order because of a hard dependency:

1. **Phase 1 first** — `Config` must have `dataset` before `build_cfg` will accept EXPERIMENTS entries that contain `"dataset"`.
2. **Phase 2** — `OASISDataset` must exist before `train.py` imports it.
3. **Phase 3** — Imports `OASISDataset`; depends on Phase 2 being done.
4. **Phase 4** — Can be done any time after Phase 1 (the key is now valid on `Config`).
5. **Phase 5** — Run only after all four phases are complete.

Phases 2 and 4 have no dependency on each other and can be done in parallel if desired.

---

## What is NOT changed

The following files require zero modifications:

| File | Reason |
|---|---|
| `train_utils.py` (`FastMRIMaskGenerator`, `simulate_undersampling`) | Works on any `[1, H, W] complex64` tensor regardless of origin |
| `normalizer.py` | Operates on the same tensor shape; OASIS fake k-space passes through identically |
| `DcTNN/` (model, loss, DC layers) | Architecture is tensor-shape agnostic |
| `train_utils.py` (`resolve_data_dirs`) | Already reads `cfg.data_dir` / `cfg.val_data_dir`; OASIS experiment overrides these keys directly in `train_config.py` |
