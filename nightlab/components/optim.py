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
    fast and good enough for the update direction. Batched over leading dims,
    as in the reference, so a (3, d, d) stack of Q, K, V gradients is
    orthogonalized slice by slice.
    """
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.to(torch.bfloat16)
    transposed = G.size(-2) > G.size(-1)
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """MomentUm Orthogonalized by Newton-schulz.

    Applies to 2D hidden weight matrices only. Reference implementation:
    https://github.com/KellerJordan/Muon

    A param group may set `chunks: k` for weights that stack k projections
    along the output dim, such as a fused qkv of shape (3*dim, dim). The update
    is then computed on the (k, rows/k, cols) view, so each projection is
    orthogonalized on its own, as the reference does for its (3, d, d) qkv.
    """

    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95, nesterov: bool = True,
                 ns_steps: int = 5, weight_decay: float = 0.0, chunks: int = 1):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
                        weight_decay=weight_decay, chunks=chunks)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, mom, wd = group["lr"], group["momentum"], group["weight_decay"]
            chunks = group["chunks"]
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
                if chunks > 1:
                    g = g.view(chunks, p.size(0) // chunks, p.size(1))
                update = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                # Scale so that updates have similar RMS regardless of matrix shape.
                update = update * max(1.0, update.size(-2) / update.size(-1)) ** 0.5
                update = update.reshape(p.shape)
                if wd:
                    p.mul_(1 - lr * wd)
                p.add_(update, alpha=-lr)


@register("optim", "muon")
def muon(model: nn.Module, cfg) -> list[torch.optim.Optimizer]:
    """Muon on hidden matrices, AdamW with the config's lr and weight decay on
    embeddings, the output head, and norms.

    Muon's own weight decay is `extra.muon_weight_decay`, default 0 like the
    reference class. It is deliberately not the AdamW value: at lr 0.02 the
    AdamW default of 0.1 would shrink hidden weights 33x faster per step than
    the baseline does."""
    hidden, other = _split_params(model)
    muon_lr = cfg.extra.get("muon_lr", 0.02)
    muon_momentum = cfg.extra.get("muon_momentum", 0.95)
    muon_wd = cfg.extra.get("muon_weight_decay", 0.0)
    stacked = {id(p) for n, p in model.named_parameters() if n.endswith("attn.qkv.weight")}
    for n, p in model.named_parameters():
        if id(p) in stacked and p.size(0) != 3 * p.size(1):
            raise ValueError(f"{n} has shape {tuple(p.shape)}, not (3*dim, dim); Muon's "
                             "per-projection split assumes a fused multi-head qkv")
    groups = [
        {"params": [p for p in hidden if id(p) not in stacked], "chunks": 1},
        {"params": [p for p in hidden if id(p) in stacked], "chunks": 3},
    ]
    groups = [g for g in groups if g["params"]]
    decay = [p for p in other if p.ndim >= 2]
    no_decay = [p for p in other if p.ndim < 2]
    fused = torch.cuda.is_available()
    return [
        Muon(groups, lr=muon_lr, momentum=muon_momentum, weight_decay=muon_wd),
        torch.optim.AdamW(
            [{"params": decay, "weight_decay": cfg.weight_decay},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=cfg.lr, betas=cfg.betas, fused=fused,
        ),
    ]
