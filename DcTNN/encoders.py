import torch
from torch import nn
from .rope_vit import compute_axial_cis, compute_mixed_cis
from .layers import TransformerEncoderLayer, TransformerEncoder
from .util import get_to_embedding, get_from_embedding, get_mlp_head


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
            self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, d_model))
            freqs_cis = None

        elif self.pos_emb_type == "Rope-Axial":
            head_dim = d_model // nhead
            freqs_cis = compute_axial_cis(dim=head_dim, end_x=grid_w, end_y=grid_h, theta=rope_theta)

        elif self.pos_emb_type == "Rope-Mixed":
            head_dim = d_model // nhead
            freqs_cis = compute_mixed_cis(dim=head_dim, end_x=grid_w, end_y=grid_h, theta=rope_theta)
        
        layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps, freqs_cis=freqs_cis)
        self.encoder = TransformerEncoder(layer, num_layers)
        
    def forward(self, img):
        x = self.to_embedding(img)
        if self.pos_emb_type == "APE":
            x = x + self.pos_embedding
        x = self.dropout(x)
        x = self.encoder(x)
        x = self.mlp_head(x)

        return x


# ---------------------------------------------------------------------------
# Encoder implementations
# ---------------------------------------------------------------------------

class TokenEncoder(BaseTokenEncoder):
    """
    Unified encoder supporting patch and kaleidoscope tokenisation strategies.
    tokenizer_type="patch"        — local N×N patch tokens
    tokenizer_type="kaleidoscope" — globally-spaced pixel tokens
    """
    def __init__(self, image_size, patch_size, numCh=1, tokenizer_type="patch", d_model=512, nhead=8,
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

        self.to_embedding = get_to_embedding(tokenizer_type, patch_height, patch_width, patch_dim, d_model)
        self.mlp_head = get_mlp_head(tokenizer_type, d_model, patch_dim, patch_height, patch_width, grid_h, numCh)

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

        self.to_horizontal_embedding, self.to_vertical_embedding = get_to_embedding(
            "axial", 
            image_height=image_height, 
            image_width=image_width, 
            numCh=numCh, 
            d_model=d_model
            )
        
        self.horizontal_mlp_head, self.vertical_mlp_head = get_mlp_head(
            "axial",
            d_model, 
            numCh=numCh, 
            image_height=image_height, 
            image_width=image_width
            )
        
        self.dropout = nn.Dropout(dropout)

        numLayers = max(num_layers // 2, 1)
        head_dim = d_model // nhead

        if pos_emb_type == "APE":
            freqs_h = None
            freqs_v = None
            self.horizontal_pos_embedding = nn.Parameter(torch.randn(1, image_width, d_model))
            self.vertical_pos_embedding = nn.Parameter(torch.randn(1, image_height, d_model))
            

        elif pos_emb_type in ("Rope-Axial", "Rope-Mixed"):
            
            freqs_h = compute_axial_cis(dim=head_dim, end_x=image_width, end_y=1, theta=rope_theta)
            freqs_v = compute_axial_cis(dim=head_dim, end_x=image_height, end_y=1, theta=rope_theta)
        
        h_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps, freqs_cis=freqs_h)
        v_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps, freqs_cis=freqs_v)
        self.horizontalEncoder = TransformerEncoder(h_layer, numLayers)
        self.verticalEncoder = TransformerEncoder(v_layer, numLayers)

    def forward(self, img):
        x = img

        x = self.to_horizontal_embedding(x)
        if self.pos_emb_type == "APE":
            x = x + self.horizontal_pos_embedding
        x = self.dropout(x)
        x = self.horizontalEncoder(x)
        x = self.horizontal_mlp_head(x)

        x = self.to_vertical_embedding(x)
        if self.pos_emb_type == "APE":
            x = x + self.vertical_pos_embedding
        x = self.dropout(x)
        x = self.verticalEncoder(x)
        x = self.vertical_mlp_head(x)
 
        return x
