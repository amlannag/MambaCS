import torch
from einops.layers.torch import Rearrange
from torch import nn
from .rope_vit import (compute_axial_cis, compute_mixed_cis, RoPEAttention, RoPETransformerEncoderLayer)


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# ---------------------------------------------------------------------------
# Base encoder for patch-like tokenisation schemes
# ---------------------------------------------------------------------------

class BaseTokenEncoder(nn.Module):
    """
    Shared base for patchEncoder and kaleidoscopeEncoder.

    Subclasses must set the following attributes in __init__ before calling
    _setup_pos_emb():
        self.pos_emb_type, self.d_model, self.nhead
        self.to_embedding, self.from_embedding, self.mlp_head, self.dropout
    """

    def _setup_pos_emb(self, grid_h, grid_w, num_patches, num_layers,
                       d_model, nhead, dim_feedforward, dropout, activation,
                       layer_norm_eps, batch_first, device, dtype, norm,
                       rope_theta, rope_mixed_rotate):
        if self.pos_emb_type == "APE":
            self.encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation,
                                           layer_norm_eps, batch_first, device, dtype),
                num_layers, norm=norm)
            self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, d_model))

        elif self.pos_emb_type == "Rope-Axial":
            head_dim = d_model // nhead
            freqs_cis = compute_axial_cis(dim=head_dim, end_x=grid_w, end_y=grid_h, theta=rope_theta)
            self.register_buffer('freqs_cis', freqs_cis)
            self.layers = nn.ModuleList([
                RoPETransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps)
                for _ in range(num_layers)
            ])

        elif self.pos_emb_type == "Rope-Mixed":
            head_dim = d_model // nhead
            freqs_cis = compute_mixed_cis(dim=head_dim, end_x=grid_w, end_y=grid_h, theta=rope_theta)
            self.register_buffer('freqs_cis', freqs_cis)
            self.layers = nn.ModuleList([
                RoPETransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps)
                for _ in range(num_layers)
            ])

        else:
            raise ValueError(f"Unknown pos_emb_type '{self.pos_emb_type}'. Choose from: APE, Rope-Axial, Rope-Mixed")

    def forward(self, img, src_mask=None):
        x = self.to_embedding(img)

        if self.pos_emb_type == "APE":
            x = x + self.pos_embedding
            x = self.dropout(x)
            x = self.encoder(x, src_mask)
        elif self.pos_emb_type in ("Rope-Axial", "Rope-Mixed"):
            x = self.dropout(x)
            freqs_cis = self.freqs_cis.to(x.device)
            for layer in self.layers:
                x = layer(x, freqs_cis)


        x = self.mlp_head(x)
        return x


# ---------------------------------------------------------------------------
# Encoder implementations
# ---------------------------------------------------------------------------

class patchEncoder(BaseTokenEncoder):
    """
    Encoder using standard patch or Kaleidoscope tokens.
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

        grid_h = image_height // patch_height
        grid_w = image_width // patch_width
        num_patches = grid_h * grid_w
        patch_dim = patch_height * patch_width * numCh

        if kaleidoscope:
            self.to_embedding = nn.Sequential(
                Rearrange('b c (k1 h) (k2 w) -> b (h w) (k1 k2 c)', k1=patch_height, k2=patch_width),
                nn.Linear(patch_dim, d_model),
            )
            self.from_embedding = Rearrange('b (h w) (k1 k2 c) -> b c (k1 h) (k2 w)',
                                            k1=patch_height, k2=patch_width, h=grid_h, c=numCh)
        else:
            self.to_embedding = nn.Sequential(
                Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
                nn.Linear(patch_dim, d_model),
            )
            self.from_embedding = Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                                            c=numCh, h=grid_h, p1=patch_height, p2=patch_width)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, patch_dim),
            self.from_embedding,
        )
        self.dropout = nn.Dropout(dropout)
        self._setup_pos_emb(grid_h, grid_w, num_patches, num_layers, d_model, nhead,
                            dim_feedforward, dropout, activation, layer_norm_eps,
                            batch_first, device, dtype, norm, rope_theta, rope_mixed_rotate)


class kaleidoscopeEncoder(BaseTokenEncoder):
    """
    Encoder using Kaleidoscope tokens — each token samples one pixel per sub-patch
    offset position across the full image grid, giving each token global spatial coverage
    from the start rather than attending to a local patch region.
    """
    def __init__(self, image_size, patch_size, numCh=1, d_model=512, nhead=8,
                num_layers=6, dim_feedforward=2048, dropout=0.1, activation='relu',
                layer_norm_eps=1e-05, batch_first=True, device=None, dtype=None, norm=None,
                pos_emb_type="APE", rope_theta=100.0, rope_mixed_rotate=True):
        super().__init__()
        self.pos_emb_type = pos_emb_type
        self.d_model = d_model
        self.nhead = nhead

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, \
            'Image dimensions must be divisible by the patch size.'

        grid_h = image_height // patch_height
        grid_w = image_width // patch_width
        num_patches = grid_h * grid_w
        patch_dim = patch_height * patch_width * numCh

        # Group by sub-patch offset: token (h,w) contains pixel (h + k1*i, w + k2*j) for all (i,j)
        self.to_embedding = nn.Sequential(
            Rearrange('b c (k1 h) (k2 w) -> b (h w) (k1 k2 c)', k1=patch_height, k2=patch_width),
            nn.Linear(patch_dim, d_model),
        )
        self.from_embedding = Rearrange('b (h w) (k1 k2 c) -> b c (k1 h) (k2 w)',
                                        k1=patch_height, k2=patch_width, h=grid_h, c=numCh)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, patch_dim),
            self.from_embedding,
        )
        self.dropout = nn.Dropout(dropout)
        self._setup_pos_emb(grid_h, grid_w, num_patches, num_layers, d_model, nhead,
                            dim_feedforward, dropout, activation, layer_norm_eps,
                            batch_first, device, dtype, norm, rope_theta, rope_mixed_rotate)


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
