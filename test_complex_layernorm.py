"""Verification tests for the 2x2 covariance ComplexLayerNorm."""
import torch
import math
import sys
sys.path.insert(0, "/Users/amlannag/Desktop/MambaCS")

from DcTNN.util import ComplexLayerNorm

torch.manual_seed(0)

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

def tag(ok): return PASS if ok else FAIL

# ── 1. Shape & dtype ──────────────────────────────────────────────────────────
d = 64
norm = ComplexLayerNorm(d)
x = torch.randn(4, 10, d, dtype=torch.cfloat)
y = norm(x)
ok_shape = (y.shape == x.shape)
ok_dtype = (y.dtype == torch.cfloat)
print(f"[{tag(ok_shape and ok_dtype)}] shape={y.shape}, dtype={y.dtype}")

# ── 2. Whitening with deliberately skewed correlated input ────────────────────
# Build input where Re and Im have unequal variance and nonzero correlation
# so the old scalar norm would NOT decorrelate.
B, T, D = 8, 512, 64
r_raw = torch.randn(B, T, D) * 3.0
i_raw = 0.5 * r_raw + torch.randn(B, T, D)
x_skew = torch.complex(r_raw, i_raw)

# Extract whitened hidden representation before gamma/beta by zeroing them
norm_check = ComplexLayerNorm(D)
with torch.no_grad():
    norm_check.gamma_rr.fill_(1.0)
    norm_check.gamma_ii.fill_(1.0)
    norm_check.gamma_ri.zero_()
    norm_check.beta.zero_()

    # Manually compute whitened output (reuse forward)
    mean = x_skew.mean(dim=-1, keepdim=True)
    centered = x_skew - mean
    r, i = centered.real, centered.imag
    V_rr = (r*r).mean(-1, keepdim=True) + norm_check.eps
    V_ii = (i*i).mean(-1, keepdim=True) + norm_check.eps
    V_ri = (r*i).mean(-1, keepdim=True)
    delta = (V_rr*V_ii - V_ri**2).clamp(min=norm_check.eps)
    s = delta.sqrt()
    t = (V_rr + V_ii + 2*s).sqrt().clamp(min=norm_check.eps)
    a = (V_ii + s) / (s * t)
    b = -V_ri / (s * t)
    d_coef = (V_rr + s) / (s * t)
    rh = a*r + b*i
    ih = b*r + d_coef*i
    white = torch.complex(rh, ih)

emp_Vrr = (white.real**2).mean(dim=-1).mean().item()
emp_Vii = (white.imag**2).mean(dim=-1).mean().item()
emp_Vri = (white.real * white.imag).mean(dim=-1).mean().item()
ok_white = (abs(emp_Vrr - 1.0) < 0.05 and abs(emp_Vii - 1.0) < 0.05 and abs(emp_Vri) < 0.05)
print(f"[{tag(ok_white)}] whitening: V_rr={emp_Vrr:.4f}, V_ii={emp_Vii:.4f}, V_ri={emp_Vri:.4f}  (want ≈1, ≈1, ≈0)")

# Confirm old scalar norm does NOT decorrelate this input
mean_old = x_skew.mean(-1, keepdim=True)
c_old = x_skew - mean_old
var_old = (c_old.abs()**2).mean(-1, keepdim=True)
w_old = c_old / (var_old + 1e-8).sqrt()
old_Vri = (w_old.real * w_old.imag).mean(dim=-1).mean().item()
ok_old_fails = abs(old_Vri) > 0.05
print(f"[{tag(ok_old_fails)}] old scalar norm V_ri={old_Vri:.4f}  (should be ≠0, confirming it doesn't decorrelate)")

# ── 3. At default init, no unwanted phase rotation on small fixed input ────────
d_small = 4
norm_init = ComplexLayerNorm(d_small)
x_fixed = torch.complex(
    torch.tensor([[1.0, -1.0, 2.0, -2.0]]),
    torch.tensor([[0.5, -0.5, 1.0, -1.0]])
)
y_fixed = norm_init(x_fixed)
# At init gamma_rr=gamma_ii=1/sqrt(2), gamma_ri=0, beta=0:
# output should be a scaled version of whitened input with no cross-mixing
inv_sqrt2 = 2**-0.5
mean_f = x_fixed.mean(-1, keepdim=True)
cf = x_fixed - mean_f
r_f, i_f = cf.real, cf.imag
Vrr_f = (r_f**2).mean(-1, keepdim=True) + 1e-8
Vii_f = (i_f**2).mean(-1, keepdim=True) + 1e-8
Vri_f = (r_f*i_f).mean(-1, keepdim=True)
delta_f = (Vrr_f*Vii_f - Vri_f**2).clamp(min=1e-8)
s_f = delta_f.sqrt()
t_f = (Vrr_f + Vii_f + 2*s_f).sqrt().clamp(min=1e-8)
af = (Vii_f + s_f)/(s_f*t_f); bf = -Vri_f/(s_f*t_f); df = (Vrr_f+s_f)/(s_f*t_f)
rh_f = af*r_f + bf*i_f; ih_f = bf*r_f + df*i_f
ref_real = inv_sqrt2 * rh_f
ref_imag = inv_sqrt2 * ih_f
ok_init = (torch.allclose(y_fixed.real, ref_real, atol=1e-5) and
           torch.allclose(y_fixed.imag, ref_imag, atol=1e-5))
print(f"[{tag(ok_init)}] init gamma produces correct scaled-identity affine (no phase rotation)")

# ── 4. Backward pass — no NaN/Inf ─────────────────────────────────────────────
x_grad = torch.randn(4, 10, 64, dtype=torch.cfloat, requires_grad=True)
norm_grad = ComplexLayerNorm(64)
y_grad = norm_grad(x_grad)
loss = y_grad.real.sum() + y_grad.imag.sum()
loss.backward()
ok_no_nan = (not torch.isnan(x_grad.grad).any() and not torch.isinf(x_grad.grad).any())
print(f"[{tag(ok_no_nan)}] backward: no NaN/Inf in input grad")

# ── 5. Degenerate near-zero-variance input (stress test clamping) ──────────────
x_degen = torch.complex(torch.zeros(2, 8, 64), torch.zeros(2, 8, 64))
x_degen = x_degen + 1e-12 * torch.randn_like(x_degen.real)  # tiny noise
norm_degen = ComplexLayerNorm(64)
try:
    y_degen = norm_degen(torch.complex(x_degen.real.cfloat().real,
                                        x_degen.real.cfloat().real))
    ok_degen = True
except Exception as e:
    ok_degen = False
    print(f"  exception: {e}")
x_degen_c = torch.complex(torch.zeros(2,8,64) + 1e-12*torch.randn(2,8,64),
                           torch.zeros(2,8,64) + 1e-12*torch.randn(2,8,64))
y_degen = norm_degen(x_degen_c)
ok_degen = not (torch.isnan(y_degen).any() or torch.isinf(y_degen).any())
print(f"[{tag(ok_degen)}] degenerate near-zero input: no NaN/Inf (clamping works)")

# ── 6. Integration: one transformer block ─────────────────────────────────────
try:
    from DcTNN.attention_layer import TransformerEncoderLayer
    block = TransformerEncoderLayer(
        d_model=64, nhead=4, dim_feedforward=128, dropout=0.0,
        activation='gelu', layer_norm_eps=1e-5,
        attn_type="complex"
    )
    x_blk = torch.randn(2, 16, 64, dtype=torch.cfloat)
    y_blk = block(x_blk)
    loss_blk = y_blk.real.sum() + y_blk.imag.sum()
    loss_blk.backward()
    ok_blk = (y_blk.shape == x_blk.shape and not torch.isnan(y_blk).any())
    print(f"[{tag(ok_blk)}] transformer block integration: shape={y_blk.shape}, no NaN")
except Exception as e:
    print(f"[{FAIL}] transformer block: {e}")

print("\nDone.")
