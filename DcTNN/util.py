from einops.layers.torch import Rearrange
from torch import nn


def get_to_embedding(tokenizer_type, patch_height=None, patch_width=None, patch_dim=None, d_model=None,
                     image_height=None, image_width=None, numCh=None):
    if tokenizer_type == "patch":
        return nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, d_model),
        )
    elif tokenizer_type == "kaleidoscope":
        return nn.Sequential(
            Rearrange('b c (k1 h) (k2 w) -> b (h w) (k1 k2 c)', k1=patch_height, k2=patch_width),
            nn.Linear(patch_dim, d_model),
        )
    elif tokenizer_type == "axial":
        return (
            nn.Sequential(
                Rearrange('b c h w -> b h (w c)'),
                nn.Linear(image_width * numCh, d_model),
            ),
            nn.Sequential(
                Rearrange('b c h w -> b w (h c)'),
                nn.Linear(image_height * numCh, d_model),
            ),
        )
    else:
        raise ValueError(f"Unknown tokenizer_type '{tokenizer_type}'. Choose from: patch, kaleidoscope, axial")


def get_from_embedding(tokenizer_type, patch_height=None, patch_width=None, grid_h=None, numCh=None,
                       image_height=None, image_width=None):
    if tokenizer_type == "patch":
        return Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                         c=numCh, h=grid_h, p1=patch_height, p2=patch_width)
    elif tokenizer_type == "kaleidoscope":
        return Rearrange('b (h w) (k1 k2 c) -> b c (k1 h) (k2 w)',
                         k1=patch_height, k2=patch_width, h=grid_h, c=numCh)
    elif tokenizer_type == "axial":
        return (
            Rearrange('b h (w c) -> b c h w', c=numCh),
            Rearrange('b w (h c) -> b c h w', c=numCh),
        )
    else:
        raise ValueError(f"Unknown tokenizer_type '{tokenizer_type}'. Choose from: patch, kaleidoscope, axial")


def get_mlp_head(tokenizer_type, d_model, patch_dim=None, patch_height=None, patch_width=None,
                 grid_h=None, numCh=None, image_height=None, image_width=None):
    if tokenizer_type in ("patch", "kaleidoscope"):
        from_emb = get_from_embedding(tokenizer_type, patch_height, patch_width, grid_h, numCh)
        return nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, patch_dim),
            from_emb,
        )
    elif tokenizer_type == "axial":
        h_from, v_from = get_from_embedding("axial", numCh=numCh)
        return (
            nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, image_width * numCh), h_from),
            nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, image_height * numCh), v_from),
        )
    else:
        raise ValueError(f"Unknown tokenizer_type '{tokenizer_type}'. Choose from: patch, kaleidoscope, axial")
