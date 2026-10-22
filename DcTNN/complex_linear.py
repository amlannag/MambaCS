"""
Complex nn.Linear with a backward pass that avoids the rocBLAS complex split-K gap.

The weight gradient of a Linear applied to T tokens is grad_y^T @ conj(x): an (out x in) output reduced over K = T.
With the strip / segment tokenisers (neighbourhood, window_axial) T is ~5e5 per batch and the output is tiny (e.g.
256 x 64). rocBLAS has no split-K kernel for complex GEMMs on gfx950, so a few workgroups reduce K serially: 112 ms
for a 68 GFLOP product on an MI355X (measured), ~100x below the device's complex-GEMM throughput. Expressing the same
product as one real fp32 GEMM over stacked [re | im] features ([K, 2*out]^T @ [K, 2*in]) runs in 0.6 ms.

ComplexLinear keeps nn.Linear's parameters and state-dict layout; only the weight-gradient computation changes.
`convert_complex_linears(model)` swaps every complex nn.Linear of a built model in place.
DCTNN_COMPLEX_LINEAR_BACKWARD=native restores the stock autograd path (A/B testing).
"""
import os

import torch
import torch.nn.functional as F
from torch import nn

_BACKENDS = ("stacked", "native")
_backend = os.environ.get("DCTNN_COMPLEX_LINEAR_BACKWARD", "stacked")
# Below this many tokens the native complex GEMM is fine and the stacking copies are not worth it.
_STACKED_MIN_TOKENS = 4096


def set_complex_linear_backend(backend: str):
    global _backend
    if backend not in _BACKENDS:
        raise ValueError(f"complex linear backend must be one of {_BACKENDS}, got {backend!r}")
    _backend = backend


def get_complex_linear_backend() -> str:
    if _backend not in _BACKENDS:
        raise ValueError(f"DCTNN_COMPLEX_LINEAR_BACKWARD must be one of {_BACKENDS}, got {_backend!r}")
    return _backend


def _stacked_weight_grad(grad_y, x):
    """grad_y^T @ conj(x) for complex grad_y [K, out], x [K, in] via one real GEMM on stacked [re | im] features."""
    out_features, in_features = grad_y.shape[1], x.shape[1]
    g = torch.cat([grad_y.real, grad_y.imag], dim=1)                     # [K, 2*out]
    xs = torch.cat([x.real, x.imag], dim=1)                              # [K, 2*in]
    r = g.T @ xs                                                         # [[gr.xr, gr.xi], [gi.xr, gi.xi]]
    gr_xr, gr_xi = r[:out_features, :in_features], r[:out_features, in_features:]
    gi_xr, gi_xi = r[out_features:, :in_features], r[out_features:, in_features:]
    return torch.complex(gr_xr + gi_xi, gi_xr - gr_xi)


class _ComplexLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias):
        ctx.save_for_backward(x, weight)
        ctx.has_bias = bias is not None
        return F.linear(x, weight, bias)

    @staticmethod
    def backward(ctx, grad_y):
        x, weight = ctx.saved_tensors
        in_features, out_features = weight.shape[1], weight.shape[0]
        g2 = grad_y.reshape(-1, out_features)
        grad_x = grad_w = grad_b = None
        if ctx.needs_input_grad[0]:
            grad_x = (g2 @ weight.conj()).reshape(x.shape)
        if ctx.needs_input_grad[1]:
            x2 = x.reshape(-1, in_features)
            grad_w = _stacked_weight_grad(g2, x2) if x2.shape[0] >= _STACKED_MIN_TOKENS else g2.T @ x2.conj()
        if ctx.has_bias and ctx.needs_input_grad[2]:
            grad_b = g2.sum(0)
        return grad_x, grad_w, grad_b


class ComplexLinear(nn.Linear):
    """nn.Linear whose backward computes the weight gradient through `_stacked_weight_grad` (complex weights only)."""

    def forward(self, x):
        if (
            get_complex_linear_backend() == "stacked"
            and self.weight.is_complex()
            and torch.is_grad_enabled()
            and (x.requires_grad or self.weight.requires_grad)
        ):
            return _ComplexLinearFunction.apply(x, self.weight, self.bias)
        return F.linear(x, self.weight, self.bias)


def convert_complex_linears(module: nn.Module) -> int:
    """Swap every plain complex nn.Linear under `module` to ComplexLinear in place; returns the number converted."""
    count = 0
    for sub in module.modules():
        if type(sub) is nn.Linear and sub.weight.is_complex():
            sub.__class__ = ComplexLinear
            count += 1
    return count
