from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import register


def rope_cache(
    seq_len: int, head_dim: int, base: float = 10000.0
) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(seq_len).float()
    freqs = torch.outer(t, inv_freq)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, H, T, D). Rotate pairs (x1, x2) -> (x1 cos - x2 sin, x1 sin + x2 cos).
    d = x.size(-1) // 2
    x1, x2 = x[..., :d], x[..., d:]
    cos, sin = cos[: x.size(2)].to(x.dtype), sin[: x.size(2)].to(x.dtype)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


@register("attention", "causal")
class CausalSelfAttention(nn.Module):
    """Plain multi-head causal attention with rotary embeddings.

    This is the baseline. It uses PyTorch's fused SDPA so that a replication
    attempt of a new attention variant competes against a fair, fast reference.
    """

    def __init__(self, dim: int, n_head: int, block_size: int, **_):
        super().__init__()
        assert dim % n_head == 0
        self.n_head = n_head
        self.head_dim = dim // n_head
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        cos, sin = rope_cache(block_size, self.head_dim)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = apply_rope(q, self.rope_cos, self.rope_sin)
        k = apply_rope(k, self.rope_cos, self.rope_sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)
