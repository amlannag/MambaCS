"""
Creates generic vision transformers.

Author: Marlon Ernesto Bran Lorenzana
Date: February 15, 2022
"""

import torch
from dc.dc import *
from einops.layers.torch import Rearrange
from torch import nn

from rope_vit import (apply_rotary_emb, compute_axial_cis, compute_mixed_cis,
                      init_random_2d_freqs, init_t_xy)

# Helpers
def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# ---------------------------------------------------------------------------
# RoPE attention and encoder layer
# ---------------------------------------------------------------------------

class RoPEAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0):
        super().__init__()
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x, freqs_cis):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.nhead, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class RoPETransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps):
        super().__init__()
        self.attn = RoPEAttention(d_model, nhead, dropout)
        act = nn.ReLU() if activation == 'relu' else nn.GELU()
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward), act,
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, freqs_cis, **kwargs):
        x = x + self.drop(self.attn(self.norm1(x), freqs_cis))
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


# ---------------------------------------------------------------------------
# ViT wrappers
# ---------------------------------------------------------------------------

class patchVIT(nn.Module):
    """
    Defines a TNN that creates either Kaleidoscope or Patch tokens.
    Args:
        N (int)                     -       Image Size
        patch_size (int)            -       Size of patch or Kaleidoscope tokens
        kaleidoscope (bool)         -       If true the network will create Kaleidoscope tokens
        layerNo (int)               -       Number of cascaded TNN
        numCh (int)                 -       Number of input channels (real, imag)
        d_model (int)               -       Model dimension
        nhead (int)                 -       Number of heads to use in multi-head attention
        num_encoder_layers (int)    -       Number of encoder layers within each encoder
        dim_feedforward (int)       -       Feedforward size in the MLP
        dropout (float)             -       Dropout of various layers
        activation                  -       Defines activation function for transformer encoder
        pos_emb_type (str)          -       "APE" | "Rope-Axial" | "Rope-Mixed"
        rope_theta (float)          -       Base frequency for RoPE
        rope_mixed_rotate (bool)    -       Randomly rotate initial freqs in Rope-Mixed
    """
    def __init__(self, N, patch_size=16, kaleidoscope=False, layerNo=2, numCh=1, d_model=None,
                    nhead=8, num_encoder_layers=2, dim_feedforward=None, dropout=0.1, activation='relu',
                    layer_norm_eps=1e-05, batch_first=True, device=None, dtype=None,
                    pos_emb_type="APE", rope_theta=100.0, rope_mixed_rotate=True):
        super(patchVIT, self).__init__()
        self.layerNo = layerNo
        self.numCh = numCh
        self.N = N
        self.patch_size = patch_size

        if d_model is None:
            d_model = patch_size * patch_size * numCh
        if dim_feedforward is None:
            dim_feedforward = int(d_model ** (3 / 2))
        self.kaleidoscope = kaleidoscope

        transformers = []
        for _ in range(layerNo):
            transformers.append(imageEncoder(self.N, self.patch_size, numCh, kaleidoscope,
                                             d_model, nhead, num_encoder_layers, dim_feedforward,
                                             dropout, activation, layer_norm_eps,
                                             batch_first, device, dtype,
                                             pos_emb_type=pos_emb_type,
                                             rope_theta=rope_theta,
                                             rope_mixed_rotate=rope_mixed_rotate))
        self.transformers = nn.ModuleList(transformers)

    def forward(self, xPrev):
        im = xPrev
        for i in range(self.layerNo):
            im = self.transformers[i](im)
        return im


class axVIT(nn.Module):
    """
    Defines the transformer for MRI reconstruction using exclusively a Transformer Encoder and axial tokens.
    Args:
            N (int)                     -       Image Size
            layerNo (int)               -       Number of cascaded TNN
            numCh (int)                 -       Number of input channels (real, imag)
            d_model (int)               -       Model dimension
            nhead (int)                 -       Number of heads to use in multi-head attention
            num_encoder_layers (int)    -       Number of encoder layers within each encoder
            dim_feedforward (int)       -       Feedforward size in the MLP
            dropout (float)             -       Dropout of various layers
            activation                  -       Defines activation function for transformer encoder
            pos_emb_type (str)          -       "APE" | "Rope-Axial" | "Rope-Mixed"
            rope_theta (float)          -       Base frequency for RoPE
            rope_mixed_rotate (bool)    -       Unused for axial encoder, kept for API consistency
    """
    def __init__(self, N, layerNo=2, numCh=1, d_model=None, nhead=8, num_encoder_layers=2,
                    dim_feedforward=None, dropout=0.1, activation='relu',
                    layer_norm_eps=1e-05, batch_first=True, device=None, dtype=None,
                    pos_emb_type="APE", rope_theta=100.0, rope_mixed_rotate=True):
        super(axVIT, self).__init__()
        self.layerNo = layerNo
        self.numCh = numCh
        self.N = N

        if d_model is None:
            d_model = N * numCh
        if dim_feedforward is None:
            dim_feedforward = int(d_model ** (3 / 2))

        transformers = []
        for _ in range(layerNo):
            transformers.append(axialEncoder(self.N, numCh, d_model, nhead, num_encoder_layers,
                                             dim_feedforward, dropout, activation, layer_norm_eps,
                                             batch_first, device, dtype,
                                             pos_emb_type=pos_emb_type,
                                             rope_theta=rope_theta))
        self.transformers = nn.ModuleList(transformers)

    def forward(self, xPrev):
        im = xPrev
        for i in range(self.layerNo):
            im = self.transformers[i](im)
        return im


class cascadeNet(nn.Module):
    """
    Defines a TNN that cascades denoising networks and applies data consistency.
    Args:
        N (int)                     -       Image Size
        encList (array)             -       Should contain denoising network
        encArgs (array)             -       Contains dictionaries with args for encoders in encList
        dcFunc (function)           -       Contains the data consistency function to be used in recon
        lamb (bool)                 -       Whether or not to use a leanred data consistency parameter
    """
    def __init__(self, N, encList, encArgs, dcFunc=FFT_DC, lamb=True):
        super(cascadeNet, self).__init__()
        if lamb:
            self.lamb = nn.Parameter(torch.ones(len(encList)) * 0.5)
        else:
            self.lamb = False
        self.N = N
        self.dcFunc = dcFunc

        transformers = []
        for i, enc in enumerate(encList):
            transformers.append(enc(N, **encArgs[i]))
        self.transformers = nn.ModuleList(transformers)

    def forward(self, xPrev, y, sampleMask):
        im = xPrev
        for i, transformer in enumerate(self.transformers):
            im_denoise = transformer(im)
            im = im + im_denoise
            if self.lamb is False:
                im = self.dcFunc(im, y, sampleMask, None)
            else:
                im = self.dcFunc(im, y, sampleMask, self.lamb[i])
        return im


# ---------------------------------------------------------------------------
# Encoder implementations
# ---------------------------------------------------------------------------

class imageEncoder(nn.Module):
    """
    Standard Encoder that utilizes image patches or kaleidoscope tokens.
    """
    def __init__(self, image_size, patch_size, numCh=1, kaleidoscope=False, d_model=512, nhead=8,
                num_layers=6, dim_feedforward=2048, dropout=0.1, activation='relu', layer_norm_eps=1e-05,
                batch_first=True, device=None, dtype=None, norm=None,
                pos_emb_type="APE", rope_theta=100.0, rope_mixed_rotate=True):
        super().__init__()

        self.pos_emb_type = pos_emb_type
        self.d_model = d_model
        self.nhead = nhead

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, \
            'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = patch_height * patch_width * numCh

        # Token embedding
        if kaleidoscope:
            self.to_embedding = nn.Sequential(
                Rearrange('b c (k1 h) (k2 w) -> b (h w) (k1 k2 c)', k1=patch_height, k2=patch_width),
                nn.Linear(patch_dim, d_model)
            )
            self.from_embedding = Rearrange('b (h w) (k1 k2 c) -> b c (k1 h) (k2 w)',
                                            k1=patch_height, k2=patch_width,
                                            h=image_height // patch_height, c=numCh)
        else:
            self.to_embedding = nn.Sequential(
                Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
                nn.Linear(patch_dim, d_model),
            )
            self.from_embedding = Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                                            c=numCh, h=image_height // patch_height,
                                            p1=patch_height, p2=patch_width)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, patch_dim),
            self.from_embedding,
        )
        self.dropout = nn.Dropout(dropout)

        grid_h = image_height // patch_height
        grid_w = image_width // patch_width

        if pos_emb_type == "APE":
            self.encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation,
                                           layer_norm_eps, batch_first, device, dtype),
                num_layers, norm=norm)
            self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, d_model))

        elif pos_emb_type == "Rope-Axial":
            head_dim = d_model // nhead
            freqs_cis = compute_axial_cis(dim=head_dim, end_x=grid_w, end_y=grid_h, theta=rope_theta)
            self.register_buffer('freqs_cis', freqs_cis)
            self.layers = nn.ModuleList([
                RoPETransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps)
                for _ in range(num_layers)
            ])

        elif pos_emb_type == "Rope-Mixed":
            head_dim = d_model // nhead
            freqs = torch.stack([
                init_random_2d_freqs(dim=head_dim, num_heads=nhead, theta=rope_theta, rotate=rope_mixed_rotate)
                for _ in range(num_layers)
            ], dim=1).view(2, num_layers, -1)
            self.freqs = nn.Parameter(freqs.clone())
            t_x, t_y = init_t_xy(end_x=grid_w, end_y=grid_h)
            self.register_buffer('freqs_t_x', t_x)
            self.register_buffer('freqs_t_y', t_y)
            self.layers = nn.ModuleList([
                RoPETransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps)
                for _ in range(num_layers)
            ])

        else:
            raise ValueError(f"Unknown pos_emb_type '{pos_emb_type}'. Choose from: APE, Rope-Axial, Rope-Mixed")

    def forward(self, img, src_mask=None):
        x = self.to_embedding(img)

        if self.pos_emb_type == "APE":
            x = x + self.pos_embedding
            x = self.dropout(x)
            x = self.encoder(x, src_mask)
        elif self.pos_emb_type == "Rope-Axial":
            x = self.dropout(x)
            freqs_cis = self.freqs_cis.to(x.device)
            for layer in self.layers:
                x = layer(x, freqs_cis)
        elif self.pos_emb_type == "Rope-Mixed":
            x = self.dropout(x)
            all_freqs = compute_mixed_cis(self.freqs, self.freqs_t_x, self.freqs_t_y, self.nhead)
            for i, layer in enumerate(self.layers):
                x = layer(x, all_freqs[i])

        x = self.mlp_head(x)
        return x


class axialEncoder(nn.Module):
    """
    Standard Encoder that utilizes axial attention (separate row and column transformers).
    """
    def __init__(self, image_size, numCh=1, d_model=512, nhead=8, num_layers=6, dim_feedforward=None,
                    dropout=0.1, activation='relu', layer_norm_eps=1e-05, batch_first=True,
                    device=None, dtype=None, norm=None,
                    pos_emb_type="APE", rope_theta=100.0):
        super().__init__()

        self.pos_emb_type = pos_emb_type
        self.d_model = d_model

        image_height, image_width = pair(image_size)

        self.to_horizontal_embedding = nn.Sequential(
            Rearrange('b c h w -> b h (w c)'),
            nn.Linear(image_width * numCh, d_model)
        )
        self.to_vertical_embedding = nn.Sequential(
            Rearrange('b c h w -> b w (h c)'),
            nn.Linear(image_height * numCh, d_model)
        )
        self.horizontal_mlp_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, image_width * numCh),
            Rearrange('b h (w c) -> b c h w', c=numCh)
        )
        self.vertical_mlp_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, image_height * numCh),
            Rearrange('b w (h c) -> b c h w', c=numCh)
        )
        self.dropout = nn.Dropout(dropout)

        numLayers = max(num_layers // 2, 1)

        if pos_emb_type == "APE":
            encoderLayer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                                      activation, layer_norm_eps, batch_first,
                                                      device, dtype)
            self.horizontalEncoder = nn.TransformerEncoder(encoderLayer, numLayers, norm=norm)
            self.verticalEncoder = nn.TransformerEncoder(encoderLayer, numLayers, norm=norm)
            self.horizontal_pos_embedding = nn.Parameter(torch.randn(1, image_width, d_model))
            self.vertical_pos_embedding = nn.Parameter(torch.randn(1, image_height, d_model))

        elif pos_emb_type in ("Rope-Axial", "Rope-Mixed"):
            head_dim = d_model // nhead
            freqs_h = compute_axial_cis(dim=head_dim, end_x=image_width, end_y=1, theta=rope_theta)
            freqs_v = compute_axial_cis(dim=head_dim, end_x=image_height, end_y=1, theta=rope_theta)
            self.register_buffer('freqs_cis_h', freqs_h)
            self.register_buffer('freqs_cis_v', freqs_v)

            def make_layer():
                return RoPETransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                                   activation, layer_norm_eps)
            self.h_layers = nn.ModuleList([make_layer() for _ in range(numLayers)])
            self.v_layers = nn.ModuleList([make_layer() for _ in range(numLayers)])

        else:
            raise ValueError(f"Unknown pos_emb_type '{pos_emb_type}'. Choose from: APE, Rope-Axial, Rope-Mixed")

    def forward(self, img, src_mask=None, src_key_padding_mask=None):
        x = img

        if self.pos_emb_type == "APE":
            x = self.to_horizontal_embedding(x)
            x = x + self.horizontal_pos_embedding
            x = self.dropout(x)
            x = self.horizontalEncoder(x, src_mask, src_key_padding_mask)
            x = self.horizontal_mlp_head(x)

            x = self.to_vertical_embedding(x)
            x = x + self.vertical_pos_embedding
            x = self.dropout(x)
            x = self.verticalEncoder(x, src_mask, src_key_padding_mask)
            x = self.vertical_mlp_head(x)

        else:
            freqs_h = self.freqs_cis_h.to(img.device)
            freqs_v = self.freqs_cis_v.to(img.device)

            x = self.to_horizontal_embedding(x)
            x = self.dropout(x)
            for layer in self.h_layers:
                x = layer(x, freqs_h)
            x = self.horizontal_mlp_head(x)

            x = self.to_vertical_embedding(x)
            x = self.dropout(x)
            for layer in self.v_layers:
                x = layer(x, freqs_v)
            x = self.vertical_mlp_head(x)

        return x
