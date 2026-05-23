"""
Defines ViT denoising blocks — cascades of encoder instances with a shared forward loop.
"""
from torch import nn
from .encoders import patchEncoder, axialEncoder, kaleidoscopeEncoder


class BaseVIT(nn.Module):
    """
    Shared base for all ViT cascade blocks.
    Subclasses build self.transformers and pass it here; forward() is shared.
    """
    def __init__(self, N, layerNo, numCh, transformers):
        super().__init__()
        self.N = N
        self.layerNo = layerNo
        self.numCh = numCh
        self.transformers = transformers

    def forward(self, xPrev):
        im = xPrev
        for transformer in self.transformers:
            im = transformer(im)
        return im


class patchVIT(BaseVIT):
    """
    Cascade of patchEncoder blocks using standard patch tokens.
    Args:
        N (int)                     -       Image Size
        patch_size (int)            -       Size of patch tokens
        kaleidoscope (bool)         -       If true, uses Kaleidoscope tokenisation inside patchEncoder
        layerNo (int)               -       Number of cascaded encoders
        numCh (int)                 -       Number of input channels
        d_model (int)               -       Model dimension (defaults to patch_size² × numCh)
        nhead (int)                 -       Number of attention heads
        num_encoder_layers (int)    -       Transformer layers per encoder
        dim_feedforward (int)       -       MLP width (defaults to d_model^1.5)
        dropout (float)             -       Dropout rate
        activation                  -       Activation function
        pos_emb_type (str)          -       "APE" | "Rope-Axial" | "Rope-Mixed"
        rope_theta (float)          -       Base frequency for RoPE
        rope_mixed_rotate (bool)    -       Random rotation for Rope-Mixed freqs
    """
    def __init__(self, N, patch_size=16, kaleidoscope=False, layerNo=2, numCh=1, d_model=None,
                    nhead=8, num_encoder_layers=2, dim_feedforward=None, dropout=0.1, activation='relu',
                    layer_norm_eps=1e-05, batch_first=True, device=None, dtype=None,
                    pos_emb_type="APE", rope_theta=100.0, rope_mixed_rotate=True):
        if d_model is None:
            d_model = patch_size * patch_size * numCh
        if dim_feedforward is None:
            dim_feedforward = int(d_model ** (3 / 2))
        transformers = nn.ModuleList([
            patchEncoder(N, patch_size, numCh, kaleidoscope, d_model, nhead, num_encoder_layers,
                         dim_feedforward, dropout, activation, layer_norm_eps, batch_first, device, dtype,
                         pos_emb_type=pos_emb_type, rope_theta=rope_theta, rope_mixed_rotate=rope_mixed_rotate)
            for _ in range(layerNo)
        ])
        super().__init__(N, layerNo, numCh, transformers)
        self.patch_size = patch_size
        self.kaleidoscope = kaleidoscope


class axVIT(BaseVIT):
    """
    Cascade of axialEncoder blocks (separate row/column transformers).
    Args:
        N (int)                     -       Image Size
        layerNo (int)               -       Number of cascaded encoders
        numCh (int)                 -       Number of input channels
        d_model (int)               -       Model dimension (defaults to N × numCh)
        nhead (int)                 -       Number of attention heads
        num_encoder_layers (int)    -       Transformer layers per encoder
        dim_feedforward (int)       -       MLP width (defaults to d_model^1.5)
        dropout (float)             -       Dropout rate
        activation                  -       Activation function
        pos_emb_type (str)          -       "APE" | "Rope-Axial" | "Rope-Mixed"
        rope_theta (float)          -       Base frequency for RoPE
        rope_mixed_rotate (bool)    -       Unused, kept for API consistency
    """
    def __init__(self, N, layerNo=2, numCh=1, d_model=None, nhead=8, num_encoder_layers=2,
                    dim_feedforward=None, dropout=0.1, activation='relu',
                    layer_norm_eps=1e-05, batch_first=True, device=None, dtype=None,
                    pos_emb_type="APE", rope_theta=100.0, rope_mixed_rotate=True):
        if d_model is None:
            d_model = N * numCh
        if dim_feedforward is None:
            dim_feedforward = int(d_model ** (3 / 2))
        transformers = nn.ModuleList([
            axialEncoder(N, numCh, d_model, nhead, num_encoder_layers, dim_feedforward,
                         dropout, activation, layer_norm_eps, batch_first, device, dtype,
                         pos_emb_type=pos_emb_type, rope_theta=rope_theta)
            for _ in range(layerNo)
        ])
        super().__init__(N, layerNo, numCh, transformers)


class kaleidoscopeVIT(BaseVIT):
    """
    Cascade of kaleidoscopeEncoder blocks — globally-sampled token tokenisation.
    Args:
        N (int)                     -       Image Size
        patch_size (int)            -       Sub-patch stride (controls token granularity)
        layerNo (int)               -       Number of cascaded encoders
        numCh (int)                 -       Number of input channels
        d_model (int)               -       Model dimension (defaults to patch_size² × numCh)
        nhead (int)                 -       Number of attention heads
        num_encoder_layers (int)    -       Encoder layers per cascade
        dim_feedforward (int)       -       MLP width (defaults to d_model^1.5)
        dropout (float)             -       Dropout rate
        activation                  -       Activation function
        pos_emb_type (str)          -       "APE" | "Rope-Axial" | "Rope-Mixed"
        rope_theta (float)          -       Base frequency for RoPE
        rope_mixed_rotate (bool)    -       Random rotation for Rope-Mixed freqs
    """
    def __init__(self, N, patch_size=16, layerNo=2, numCh=1, d_model=None,
                    nhead=8, num_encoder_layers=2, dim_feedforward=None, dropout=0.1,
                    activation='relu', layer_norm_eps=1e-05, batch_first=True,
                    device=None, dtype=None,
                    pos_emb_type="APE", rope_theta=100.0, rope_mixed_rotate=True):
        if d_model is None:
            d_model = patch_size * patch_size * numCh
        if dim_feedforward is None:
            dim_feedforward = int(d_model ** (3 / 2))
        transformers = nn.ModuleList([
            kaleidoscopeEncoder(N, patch_size, numCh, d_model, nhead, num_encoder_layers,
                                dim_feedforward, dropout, activation, layer_norm_eps,
                                batch_first, device, dtype,
                                pos_emb_type=pos_emb_type, rope_theta=rope_theta,
                                rope_mixed_rotate=rope_mixed_rotate)
            for _ in range(layerNo)
        ])
        super().__init__(N, layerNo, numCh, transformers)
        self.patch_size = patch_size
