# Plan 01 — Non-square patches (16×32) + axial row-stride=2

## Goal
For a 640×320 MRI image:
- **Patch / Kaleidoscope**: use 16 (H) × 32 (W) tokens → grid 40×10 = 400 tokens
- **Axial horizontal**: group 2 rows per token → 640/2 = 320 tokens (matches 320 vertical tokens)
- Both axial directions now produce 320 tokens at `d_model = image_width * numCh = 320`

---

## Phase 0: Findings Summary (already gathered)

### Allowed APIs / key signatures

| File | Symbol | Signature / note |
|---|---|---|
| `encoders.py:9` | `pair()` | Already handles tuples — `pair((16,32))` → `(16,32)` |
| `encoders.py:84–85` | TokenEncoder init | `patch_height, patch_width = pair(patch_size)` — already safe |
| `vit.py:51` | TokenVIT d_model | **Bug**: `patch_size * patch_size * numCh` — scalar-only |
| `vit.py:88` | axVIT d_model | `d_model = image_width * numCh` — unchanged |
| `util.py:96–100` | `get_to_embedding` patch | `Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=ph, p2=pw)` |
| `util.py:101–105` | `get_to_embedding` kaleido | `Rearrange('b c (k1 h) (k2 w) -> b (h w) (k1 k2 c)', k1=ph, k2=pw)` |
| `util.py:106–116` | `get_to_embedding` axial | returns tuple; horiz: `'b c h w -> b h (w c)'`; vert: `'b c h w -> b w (h c)'` |
| `encoders.py:142–143` | axialEncoder APE | **Bug**: `horizontal_pos_embedding` uses `image_width`, should be `image_height` (swapped) — irrelevant currently since both experiments use Rope-Axial, but fix while here |
| `encoders.py:146–147` | axialEncoder RoPE | `freqs_h = cis_fn(end_x=image_height, end_y=1)` — needs `// row_stride` |
| `train_utils.py:24–38` | `_ENCODER_ARGS["axial"]` | builds `axVIT` kwargs dict — add `row_stride` here |

### Anti-patterns
- Do NOT change `pair()` — it already handles tuples correctly
- Do NOT change kaleidoscope Rearrange pattern — it re-uses `patch_height/width` which will naturally become `(16, 32)`
- The axial horizontal APE pos embedding dimension is `image_width` in the code but the token count is `image_height` — this is a latent bug (masked because all experiments use Rope-Axial); fix it in this pass

---

## Phase 1 — Config: expose the two new knobs

**Files**: `config.py`

### Tasks

1. Change `patch_size` default from scalar `16` to tuple `(16, 32)`:
   ```python
   # line 72 — was: patch_size: int = 16
   patch_size: tuple = (16, 32)
   ```

2. Add `axial_row_stride` field (default 2):
   ```python
   axial_row_stride: int = 2
   ```

### Verification
- `grep -n "patch_size" config.py` → shows tuple default
- `grep -n "axial_row_stride" config.py` → shows new field

---

## Phase 2 — vit.py: fix square d_model assumption in TokenVIT; thread row_stride into axVIT

**File**: `DcTNN/vit.py`

### 2a — TokenVIT (line 51)

The default `d_model` calculation assumes square patches:
```python
# BEFORE (line 51):
d_model = patch_size * patch_size * numCh

# AFTER — import pair from encoders or define locally:
from .encoders import pair          # pair already defined there
ph, pw = pair(patch_size)
d_model = ph * pw * numCh
```

For `patch_size=(16,32)`, `numCh=1`: `d_model = 16*32 = 512`.

### 2b — axVIT: add row_stride param and pass down

```python
# BEFORE signature (line 83):
def __init__(self, N, layerNo=2, numCh=1, d_model=None, nhead=8, ...):

# AFTER:
def __init__(self, N, layerNo=2, numCh=1, d_model=None, nhead=8, ..., row_stride=1):
```

Pass `row_stride=row_stride` to each `axialEncoder(...)` instantiation inside the list comprehension (lines 92–96).

### Verification
- `python -c "from DcTNN.vit import TokenVIT, axVIT; print('ok')"` — no import errors
- `TokenVIT(N=(640,320), patch_size=(16,32), numCh=1)` — d_model should print 512

---

## Phase 3 — encoders.py: thread row_stride into axialEncoder

**File**: `DcTNN/encoders.py`

### Tasks

1. Add `row_stride: int = 1` to `axialEncoder.__init__` signature (line 112).

2. Compute the effective horizontal token count:
   ```python
   h_tokens = image_height // row_stride   # 640 // 2 = 320
   ```

3. Pass `row_stride` to `get_to_embedding` and `get_mlp_head`:
   ```python
   self.to_horizontal_embedding, self.to_vertical_embedding = get_to_embedding(
       "axial", image_height=image_height, image_width=image_width,
       numCh=numCh, d_model=d_model, row_stride=row_stride,
       is_complex=self.is_complex)

   self.horizontal_mlp_head, self.vertical_mlp_head = get_mlp_head(
       "axial", d_model, numCh=numCh, image_height=image_height,
       image_width=image_width, row_stride=row_stride,
       is_complex=self.is_complex)
   ```

4. Fix APE positional embedding shapes (latent bug — was swapped):
   ```python
   # BEFORE (lines 142–143):
   self.horizontal_pos_embedding = nn.Parameter(torch.randn(1, image_width,  d_model, ...))
   self.vertical_pos_embedding   = nn.Parameter(torch.randn(1, image_height, d_model, ...))

   # AFTER:
   self.horizontal_pos_embedding = nn.Parameter(torch.randn(1, h_tokens,     d_model, ...))
   self.vertical_pos_embedding   = nn.Parameter(torch.randn(1, image_width,  d_model, ...))
   ```

5. Fix RoPE frequencies for horizontal path:
   ```python
   # BEFORE (line 146):
   freqs_h = cis_fn(dim=head_dim, end_x=image_height, end_y=1, theta=rope_theta)

   # AFTER:
   freqs_h = cis_fn(dim=head_dim, end_x=h_tokens, end_y=1, theta=rope_theta)
   ```

### Verification
- `axialEncoder((640,320), row_stride=2)` constructs without error
- Horizontal pos embedding shape: `(1, 320, d_model)` ✓
- Vertical pos embedding shape: `(1, 320, d_model)` ✓

---

## Phase 4 — util.py: new Rearrange patterns for row-strided axial

**File**: `DcTNN/util.py`

### 4a — `get_to_embedding` (lines 106–116)

Add `row_stride=1` parameter. Update horizontal path:

```python
# BEFORE horizontal path:
nn.Sequential(
    Rearrange('b c h w -> b h (w c)'),
    nn.Linear(image_width * numCh, d_model, dtype=dtype),
)

# AFTER (row_stride groups p rows into each token):
nn.Sequential(
    Rearrange('b c (h p) w -> b h (p w c)', p=row_stride),
    nn.Linear(row_stride * image_width * numCh, d_model, dtype=dtype),
)
```

Vertical path is unchanged.

Note: when `row_stride=1`, `Rearrange('b c (h p) w -> b h (p w c)', p=1)` is equivalent to the original.

### 4b — `get_from_embedding` (lines 121–135)

Add `row_stride=1` parameter. Update horizontal reverse path:

```python
# BEFORE horizontal from_embedding:
Rearrange('b h (w c) -> b c h w', w=image_width, c=numCh)

# AFTER:
Rearrange('b h (p w c) -> b c (h p) w', p=row_stride, w=image_width, c=numCh)
```

Vertical path is unchanged.

### 4c — `get_mlp_head` (lines 138–156)

Add `row_stride=1` parameter. Pass through to `get_from_embedding` and update horizontal output linear:

```python
# BEFORE (axial horizontal output):
nn.Sequential(norm(d_model), nn.Linear(d_model, image_width * numCh, dtype=dtype), h_from)

# AFTER:
nn.Sequential(norm(d_model), nn.Linear(d_model, row_stride * image_width * numCh, dtype=dtype), h_from)
```

And `h_from = get_from_embedding("axial", numCh=numCh, image_width=image_width, row_stride=row_stride)`.

### Verification
- Trace a dummy tensor `(2, 1, 640, 320)` through `to_horizontal_embedding` → shape should be `(2, 320, d_model)` ✓
- Trace through `horizontal_mlp_head` → shape should be `(2, 1, 640, 320)` ✓

---

## Phase 5 — train_utils.py: expose row_stride in _ENCODER_ARGS

**File**: `train_utils.py`

Add `row_stride=cfg.axial_row_stride` to the `"axial"` entry in `_ENCODER_ARGS`:

```python
"axial": lambda cfg: (
    axVIT,
    dict(
        ...
        row_stride=cfg.axial_row_stride,   # ← add this line
    ),
),
```

### Verification
- `grep "row_stride" train_utils.py` → one hit in the axial dict

---

## Phase 6 — Smoke test with build_model.py

Run the existing sanity check:
```bash
python build_model.py
```

Expected:
- No shape errors during forward pass on the Shepp-Logan phantom
- Parameter count printed successfully

If the phantom is square (320×320), also manually test with a `(1,1,640,320)` tensor through `axVIT(N=(640,320), row_stride=2)` to confirm 320-token output.

---

## Change Summary Table

| File | Change | Reason |
|---|---|---|
| `config.py` | `patch_size=(16,32)`, add `axial_row_stride=2` | expose both knobs |
| `DcTNN/vit.py` | TokenVIT d_model uses `pair(patch_size)` | fix square assumption |
| `DcTNN/vit.py` | axVIT accepts + passes `row_stride` | thread param down |
| `DcTNN/encoders.py` | axialEncoder accepts `row_stride`, uses `h_tokens`, fixes APE/RoPE shapes | core logic |
| `DcTNN/util.py` | `get_to_embedding/from_embedding/mlp_head` updated for `row_stride` | Rearrange patterns |
| `train_utils.py` | axial encoder args includes `row_stride` | wire to config |
