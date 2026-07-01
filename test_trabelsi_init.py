"""Verification tests for Trabelsi complex weight initialization."""
import sys, math
sys.path.insert(0, "/Users/amlannag/Desktop/MambaCS")

import numpy as np
import torch
import torch.nn as nn

from DcTNN.complex_init import trabelsi_init_, apply_trabelsi_, print_complex_init_summary

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
def tag(ok): return PASS if ok else FAIL

np.random.seed(42); torch.manual_seed(42)

# ── 1. He: E[|W|^2] ≈ 2/fan_in ───────────────────────────────────────────────
fan_in = 512
w_he = nn.Parameter(torch.empty(fan_in, fan_in, dtype=torch.cfloat))
trabelsi_init_(w_he, fan_in=fan_in, criterion="he")
emp = w_he.abs().pow(2).mean().item()
expected = 2.0 / fan_in
ok_he = abs(emp - expected) / expected < 0.05
print(f"[{tag(ok_he)}] He:     E[|W|^2]={emp:.6f}  expected={expected:.6f}  ({100*abs(emp-expected)/expected:.1f}% error)")

# ── 2. Glorot: E[|W|^2] ≈ 2/(fan_in+fan_out) ─────────────────────────────────
fan_in, fan_out = 256, 512
w_gl = nn.Parameter(torch.empty(fan_out, fan_in, dtype=torch.cfloat))
trabelsi_init_(w_gl, fan_in=fan_in, fan_out=fan_out, criterion="glorot")
emp = w_gl.abs().pow(2).mean().item()
expected = 2.0 / (fan_in + fan_out)
ok_gl = abs(emp - expected) / expected < 0.05
print(f"[{tag(ok_gl)}] Glorot: E[|W|^2]={emp:.6f}  expected={expected:.6f}  ({100*abs(emp-expected)/expected:.1f}% error)")

# ── 3. Phase distribution ≈ Uniform(-π, π) ────────────────────────────────────
phases = torch.angle(w_gl).flatten().detach().numpy()
# For uniform(-pi,pi), mean≈0 and std≈pi/sqrt(3)≈1.814
phase_mean = phases.mean()
phase_std  = phases.std()
expected_std = math.pi / math.sqrt(3)
ok_phase = abs(phase_mean) < 0.1 and abs(phase_std - expected_std) / expected_std < 0.05
print(f"[{tag(ok_phase)}] Phase: mean={phase_mean:.4f} (want≈0), std={phase_std:.4f} (want≈{expected_std:.4f})")

# ── 4. Biases zeroed ──────────────────────────────────────────────────────────
layer = nn.Linear(64, 128, dtype=torch.cfloat)
apply_trabelsi_(layer, criterion="glorot")
ok_bias = layer.bias is not None and layer.bias.abs().max().item() == 0.0
print(f"[{tag(ok_bias)}] Bias:  max|bias|={layer.bias.abs().max().item()}")

# ── 5. Forward pass through full model — no NaN/Inf ───────────────────────────
from DcTNN.attention_layer import TransformerEncoderLayer
block = TransformerEncoderLayer(
    d_model=64, nhead=4, dim_feedforward=128, dropout=0.0,
    activation='gelu', layer_norm_eps=1e-5, attn_type="complex"
)
x = torch.randn(2, 16, 64, dtype=torch.cfloat)
y = block(x)
ok_fwd = y.shape == x.shape and not torch.isnan(y).any() and not torch.isinf(y).any()
print(f"[{tag(ok_fwd)}] Forward: shape={y.shape}, no NaN/Inf")

# ── 6. Summary table ──────────────────────────────────────────────────────────
print("\n--- Summary table for single TransformerEncoderLayer ---")
print_complex_init_summary(block)

# ── 7. Full model summary (vit) ───────────────────────────────────────────────
try:
    from DcTNN.vit import axVIT
    model = axVIT(
        N=(320, 320), layerNo=2, numCh=1,
        d_model=64, nhead=4, num_encoder_layers=2, dim_feedforward=128,
        dropout=0.0, attn_type="complex"
    )
    x_img = torch.randn(1, 1, 320, 320, dtype=torch.cfloat)
    out = model(x_img)
    ok_model = not torch.isnan(out).any() and not torch.isinf(out).any()
    print(f"[{tag(ok_model)}] Full axVIT forward: output shape={out.shape}, no NaN/Inf")
    print("\n--- Summary table for axVIT ---")
    print_complex_init_summary(model)
except Exception as e:
    print(f"[{FAIL}] axVIT test: {e}")

print("\nDone.")
