from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import register


@register("mlp", "swiglu")
class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_mult: float = 8 / 3, **_):
        super().__init__()
        hidden = int(dim * hidden_mult)
        hidden = (hidden + 63) // 64 * 64
        self.w_gate = nn.Linear(dim, hidden, bias=False)
        self.w_up = nn.Linear(dim, hidden, bias=False)
        self.w_down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


@register("mlp", "gelu")
class GELUMLP(nn.Module):
    def __init__(self, dim: int, hidden_mult: float = 4.0, **_):
        super().__init__()
        hidden = int(dim * hidden_mult)
        self.w_up = nn.Linear(dim, hidden, bias=False)
        self.w_down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.gelu(self.w_up(x), approximate="tanh"))


@register("mlp", "relu2")
class ReLUSquaredMLP(nn.Module):
    """ReLU^2, as used in the modded-nanogpt speedruns."""

    def __init__(self, dim: int, hidden_mult: float = 4.0, **_):
        super().__init__()
        hidden = int(dim * hidden_mult)
        self.w_up = nn.Linear(dim, hidden, bias=False)
        self.w_down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.relu(self.w_up(x)).square())
