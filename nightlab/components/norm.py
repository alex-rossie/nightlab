from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import register


@register("norm", "rmsnorm")
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, **_):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (x.size(-1),), self.weight, self.eps)


@register("norm", "layernorm")
class LayerNorm(nn.LayerNorm):
    def __init__(self, dim: int, eps: float = 1e-5, **_):
        super().__init__(dim, eps=eps)
