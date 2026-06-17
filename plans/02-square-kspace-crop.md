# Plan 02 — Square K-Space Center Crop (640×320 → 320×320)

## Goal
Change data preprocessing so k-space slices are center-cropped to 320×320 (square) instead
of 640×320. Currently only columns are cropped; this plan also crops the center 320 rows.

---

## Phase 0: Documentation Discovery (Complete)

### Allowed APIs / Confirmed Facts
- `center_crop_kspace(kspace, N)` in `dataset.py:8` — currently crops columns only
- `H5MRIDataset(data_dir, N=320, ...)` in `dataset.py:27` — `N` is width only
- `config.py:46` — `image_size: Tuple[int, int] = (640, 320)` is the primary shape config
- `train.py:162` — extracts `img_w = cfg.image_size[1]`, passes `N=img_w` to dataset
- `inference.py:167-176` — same pattern: extracts `img_h, img_w`, passes `N=img_w`
- `kspace_analysis.ipynb` cell 3 — creates `ks_tensor` from raw `ks` (640×372), no height crop
- `build_model.py:10` — `N = 320` (square test, already correct, no change needed)
- No hardcoded 640 in DcTNN model code — shape flows through config dynamically

### Anti-Patterns to Avoid
- Don't change model code (DcTNN/) — it accepts `image_size` dynamically
- Don't add a separate `H` parameter to `H5MRIDataset` — use `image_size` tuple consistently
- Don't break the `N=img_w` pattern in train.py without updating inference.py too

---

## Phase 1: `dataset.py` — Add Height Crop

**What to change:**
- Rename `center_crop_kspace(kspace, N)` → `center_crop_kspace(kspace, image_size)`
  - Unpack `image_size` into `(H, W)`
  - Crop rows: `h0 = (kspace.shape[0] - H) // 2; kspace[h0:h0+H, ...]`
  - Crop cols: `w0 = (kspace.shape[1] - W) // 2; kspace[:, w0:w0+W]`
  - Combined: `kspace[h0:h0+H, w0:w0+W]`
- Update `H5MRIDataset.__init__` signature: `N=320` → `image_size=(320, 320)`
  - Store as `self.image_size = image_size`
- Update `__getitem__`: call `center_crop_kspace(kspace, self.image_size)`
- Update docstrings: `[1, H, N]` → `[1, image_size[0], image_size[1]]`

**Verification:**
```python
from dataset import H5MRIDataset
ds = H5MRIDataset("path/to/data", image_size=(320, 320))
sample = ds[0]
assert sample.shape == (1, 320, 320), sample.shape
```

---

## Phase 2: `config.py` — Update Default Image Size

**What to change:**
- Line 46: `image_size: Tuple[int, int] = (640, 320)` → `(320, 320)`

**Verification:**
```python
from config import Config
assert Config().image_size == (320, 320)
```

---

## Phase 3: `train.py` — Pass `image_size` to Dataset

**What to change:**
- Line 162: Remove the `_, img_w = cfg.image_size ...` extraction for dataset construction
- Lines 167-171: Change `H5MRIDataset(train_data_dir, N=img_w, ...)` 
  → `H5MRIDataset(train_data_dir, image_size=cfg.image_size, ...)`
- Keep `img_w` extraction only for the print statement on line 163 (acceleration display)

**Verification:**
- Grep: no remaining `N=img_w` in train.py

---

## Phase 4: `inference.py` — Pass `image_size` to Dataset

**What to change:**
- Lines 167-176: Change `H5MRIDataset(..., N=img_w, ...)` → `H5MRIDataset(..., image_size=image_size, ...)`
  where `image_size` is already extracted as `data_cfg["image_size"]`

**Verification:**
- Grep: no remaining `N=img_w` in inference.py

---

## Phase 5: `kspace_analysis.ipynb` — Center-Crop Height in Notebook

**What to change (cell 3, id `38d154c0`):**
```python
# Before — uses raw ks (640, 372):
mask = generate_column_mask(ks.shape, accel=4, device=torch.device("cpu"))
ks_tensor = torch.from_numpy(ks).unsqueeze(0).unsqueeze(0)  # (1, 1, 640, 372)

# After — crop to 320×320 first:
TARGET_H, TARGET_W = 320, 320
h, w = ks.shape
h0 = (h - TARGET_H) // 2
w0 = (w - TARGET_W) // 2
ks_sq = ks[h0:h0+TARGET_H, w0:w0+TARGET_W]  # (320, 320)

mask = generate_column_mask((TARGET_H, TARGET_W), accel=8, device=torch.device("cpu"))
ks_tensor = torch.from_numpy(ks_sq).unsqueeze(0).unsqueeze(0)  # (1, 1, 320, 320)
```

**Verification:**
- `ks_tensor.shape == (1, 1, 320, 320)`
- `mask.shape == (320, 320)`

---

## Phase 6: Final Verification

```bash
# No remaining 640 shape references
grep -rn "640" --include="*.py" /path/to/MambaCS/ | grep -v ".pyc"

# Sanity check model still builds
python build_model.py
```

**Expected:** `build_model.py` passes, no shape errors. All assertions in phases 1–5 pass.
