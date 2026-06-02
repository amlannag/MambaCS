import copy
import torch.nn as nn
from .rope_vit import apply_rotary_emb


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0, freqs_cis=None):
        super().__init__()
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.use_rope = freqs_cis is not None
        if self.use_rope:
            self.register_buffer('freqs_cis', freqs_cis)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.nhead, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.use_rope:
            q, k = apply_rotary_emb(q, k, freqs_cis=self.freqs_cis.to(x.device))
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps, freqs_cis=None):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, nhead, dropout, freqs_cis=freqs_cis)
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

    def forward(self, x):
        x = x + self.drop(self.attn(self.norm1(x)))
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


class TransformerEncoder(nn.Sequential):
    def __init__(self, encoder_layer, num_layers):
        super().__init__(*[copy.deepcopy(encoder_layer) for _ in range(num_layers)])
