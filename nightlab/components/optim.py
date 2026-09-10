"""Optimizers.

Each entry is a factory: (model, cfg) -> list[torch.optim.Optimizer].
Returning a list lets an experiment mix optimizers, for example Muon on the
hidden matrices and AdamW on embeddings, norms, and the output head.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from . import register


def _split_params(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Hidden 2D weights vs everything else (embeddings, head, norms, biases)."""
    hidden, other = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_matrix = p.ndim == 2
        is_embed_or_head = name.startswith(("tok_emb", "lm_head"))
        (hidden if is_matrix and not is_embed_or_head else other).append(p)
    return hidden, other


@register("optim", "adamw")
def adamw(model: nn.Module, cfg) -> list[torch.optim.Optimizer]:
    hidden, other = _split_params(model)
    decay = [p for p in hidden + other if p.ndim >= 2]
    no_decay = [p for p in hidden + other if p.ndim < 2]
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    fused = torch.cuda.is_available()
    return [torch.optim.AdamW(groups, lr=cfg.lr, betas=cfg.betas, fused=fused)]


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Approximate orthogonalization via a quintic Newton-Schulz iteration.

    Coefficients from Keller Jordan's Muon. The iteration is run in bfloat16
    and does not converge to exactly orthogonal, which is intentional: it is
    fast and good enough for the update direction.
    """
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.to(torch.bfloat16)
    X = X / (X.norm() + eps)
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """MomentUm Orthogonalized by Newton-schulz.

    Applies to 2D hidden weight matrices only. Reference implementation:
    https://github.com/KellerJordan/Muon
    """

    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95, nesterov: bool = True,
                 ns_steps: int = 5, weight_decay: float = 0.0):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
                        weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, mom, wd = group["lr"], group["momentum"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(mom).add_(g)
                if group["nesterov"]:
                    g = g.add(buf, alpha=mom)
                else:
                    g = buf
                update = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                # Scale so that updates have similar RMS regardless of matrix shape.
                update = update * max(1.0, p.size(0) / p.size(1)) ** 0.5
                if wd:
                    p.mul_(1 - lr * wd)
                p.add_(update, alpha=-lr)


@register("optim", "muon")
def muon(model: nn.Module, cfg) -> list[torch.optim.Optimizer]:
    hidden, other = _split_params(model)
    muon_lr = cfg.extra.get("muon_lr", 0.02)
    muon_momentum = cfg.extra.get("muon_momentum", 0.95)
    decay = [p for p in other if p.ndim >= 2]
    no_decay = [p for p in other if p.ndim < 2]
    fused = torch.cuda.is_available()
    return [
        Muon(hidden, lr=muon_lr, momentum=muon_momentum, weight_decay=cfg.weight_decay),
        torch.optim.AdamW(
            [{"params": decay, "weight_decay": cfg.weight_decay},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=cfg.lr, betas=cfg.betas, fused=fused,
        ),
    ]
