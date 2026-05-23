"""
RoPE utility functions for Vision Transformers.

Adapted from:
https://github.com/naver-ai/rope-vit
https://github.com/meta-llama/codellama/blob/main/llama/model.py
"""

import torch
import torch.nn as nn

def init_t_xy(end_x: int, end_y: int):
    """
    Initialize 2D coordinate tensors for RoPE.
    Args:
        end_x (int): The size of the x-dimension (width).
        end_y (int): The size of the y-dimension (height).
    Returns:
        t_x (torch.Tensor): A tensor of shape (end_x * end_y,) containing the x-coordinates.
        t_y (torch.Tensor): A tensor of shape (end_x * end_y,) containing the y-coordinates.
    """
    # Constructs a linear [0,1,2,...,end_x*end_y-1] and maps to 2D coordinates
    t = torch.arange(end_x * end_y, dtype=torch.float32)
    # tx is the col of the data [0,1,2,...,end_x-1,0,1,2,...,end_x-1,...]
    t_x = (t % end_x).float()
    # ty is the row of the data [0,0,...,0,1,1,...,1,...]
    t_y = torch.div(t, end_x, rounding_mode='floor').float()
    # A grid of x and y cordinates as 2 lists
    return t_x, t_y


def compute_axial_cis(dim: int, end_x: int, end_y: int, theta: float = 100.0):
    """
    Compute axial RoPE frequencies for 2D inputs.
    Args:
    dim (int): The dimension of the model (must be divisible by 4).
    end_x (int): The size of the x-dimension (width).
    end_y (int): The size of the y-dimension (height).
    theta (float): The theta base for RoPE
    Returns:
    freqs_cis (torch.Tensor): A tensor of shape (end_x * end_y, dim) containing the RoPE frequencies for each position.
    """
    # freq = theta ^ {-d/dim} where d is the dimension inside d_model
    # [:dim//4] because we only one to let the first dim//4 through
    freq = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    # retrieve the 2D coordinates for each position in the input
    t_x, t_y = init_t_xy(end_x, end_y)
    # the matrix contains e^{i * freq * t_x}. Outer product gives us the total frequencies
    freqs_x = torch.outer(t_x, freq)
    freqs_y = torch.outer(t_y, freq)
    # torch.ones_like creates a magnituide 1 tensor with the same shape 
    # torch.polar converts it in to imaginary as e^{i*theta} = cos(theta) + i*sin(theta)
    freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x)
    freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y)
    # Outputs a matrix which has sin and cos interleaved for x and y dimensions [cos_x1, sin_x1, cos_x2, sin_x2, ..., cos_y1, sin_y1, cos_y2, sin_y2, ...]
    return torch.cat([freqs_cis_x, freqs_cis_y], dim=-1)


def compute_mixed_cis(dim: int, end_x: int, end_y: int, theta: float = 100.0):
    """
    Compute mixed RoPE frequencies for 2D inputs.
    Like compute_axial_cis but adds x and y frequencies instead of concatenating,
    producing a single rotation per position of shape (end_x*end_y, dim//2).
    """
    freq = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t_x, t_y = init_t_xy(end_x, end_y)
    freqs_x = torch.outer(t_x, freq)
    freqs_y = torch.outer(t_y, freq)
    return torch.polar(torch.ones_like(freqs_x), freqs_x + freqs_y)



def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Reshape the RoPE frequencies for broadcasting with the input tensors.
    Args:
    freqs_cis (torch.Tensor): A tensor containing the RoPE frequencies.
    x (torch.Tensor): The input tensor to which the RoPE frequencies will be applied.
    Returns:
    freqs_cis (torch.Tensor): The reshaped RoPE frequencies ready for broadcasting with the input tensor.
    """
    # Number of dims of x (batch_size, num_heads, seq_len, head_dim) has 4 dims
    ndim = x.ndim
    assert 1 < ndim
    if freqs_cis.shape == (x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim - 2 else 1 for i, d in enumerate(x.shape)]
    elif freqs_cis.shape == (x.shape[-3], x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim - 3 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    """
    Apply RoPE to the query and key tensors.
    Args:
    xq (torch.Tensor): The query tensor of shape (batch_size, num_heads, seq_len, head_dim).
    xk (torch.Tensor): The key tensor of shape (batch_size, num_heads, seq_len, head_dim).
    freqs_cis (torch.Tensor): The RoPE frequencies of shape (num_heads, seq_len, head_dim) or (seq_len, head_dim).
    Returns:
    xq_out (torch.Tensor): The query tensor after applying RoPE, of the same shape as xq.
    xk_out (torch.Tensor): The key tensor after applying RoPE, of the same shape as xk.
    """
    # Folds xq and xk into half and treats them as complex numbers
    # xq is (batch_size, num_heads, seq_len, head_dim) -> (batch_size, num_heads, seq_len, head_dim//2) with complex numbers
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    # Reshape freqs_cis for broadcasting with xq and xk
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    # Rotates xq and xk and unbfolds them back to the original shape
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    
    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)




# ---------------------------------------------------------------------------
# RoPE-based attention and transformer encoder layer
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
