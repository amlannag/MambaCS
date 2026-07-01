import math
import numpy as np
import torch
import torch.nn as nn


def trabelsi_init_(weight: torch.Tensor, fan_in: int, fan_out: int = None, criterion: str = "he"):
    """
    Trabelsi et al. (ICLR 2018) complex weight initialization.
    Draws magnitude ~ Rayleigh(sigma), phase ~ Uniform(-pi, pi).

    criterion="he":     Var(W) = 2/fan_in       (before rectifying activations: CReLU, CGELU)
    criterion="glorot": Var(W) = 2/(fan_in+fan_out)  (Q/K/V/O projections, FFN output, embeddings)

    # TODO: Trabelsi et al. also describe a semi-unitary "independent" initialization
    # variant (reshaped from a random unitary matrix, then rescaled to target variance)
    # used in all their reported experiments. Implement this as a fallback if training
    # instability is observed with the standard Rayleigh/uniform-phase variant.
    """
    if criterion in ("glorot", "xavier"):
        assert fan_out is not None, "fan_out required for glorot criterion"
        sigma = 1.0 / math.sqrt(fan_in + fan_out)
    elif criterion in ("he", "kaiming"):
        sigma = 1.0 / math.sqrt(fan_in)
    else:
        raise ValueError(f"Unknown criterion: {criterion!r}")

    shape = tuple(weight.shape)
    rho = np.random.rayleigh(scale=sigma, size=shape)
    theta = np.random.uniform(-np.pi, np.pi, size=shape)

    with torch.no_grad():
        if torch.is_complex(weight):
            weight.copy_(
                torch.from_numpy(rho * np.cos(theta) + 1j * rho * np.sin(theta)).to(weight.dtype)
            )
        else:
            raise TypeError(
                "weight must be a complex tensor; if using separate real/imag Parameters, "
                "apply to each component's combined (real, imag) pair manually."
            )
    return weight


def apply_trabelsi_(layer: nn.Linear, criterion: str = "glorot"):
    """Apply Trabelsi init to a complex nn.Linear and zero its bias."""
    # nn.Linear weight layout: (out_features, in_features)
    fan_out, fan_in = layer.weight.shape[0], layer.weight.shape[1]
    trabelsi_init_(layer.weight, fan_in=fan_in, fan_out=fan_out, criterion=criterion)
    if layer.bias is not None:
        with torch.no_grad():
            layer.bias.zero_()
    layer._trabelsi_criterion = criterion


def print_complex_init_summary(model: nn.Module):
    """Print all complex linear layers tagged with a Trabelsi init criterion."""
    rows = [(n, tuple(m.weight.shape), m._trabelsi_criterion)
            for n, m in model.named_modules()
            if isinstance(m, nn.Linear) and hasattr(m, "_trabelsi_criterion")]
    if not rows:
        print("No Trabelsi-initialized layers found.")
        return
    print(f"\n{'Layer':<55} {'Weight shape':<22} Criterion")
    print("-" * 85)
    for name, shape, crit in rows:
        print(f"{name:<55} {str(shape):<22} {crit}")
    print()
