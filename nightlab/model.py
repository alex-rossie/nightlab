from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import components
from .config import ModelConfig


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        Norm = components.get("norm", cfg.norm)
        Attn = components.get("attention", cfg.attention)
        MLP = components.get("mlp", cfg.mlp)
        self.norm1 = Norm(cfg.n_embd, **cfg.extra)
        self.attn = Attn(cfg.n_embd, n_head=cfg.n_head, block_size=cfg.block_size, **cfg.extra)
        self.norm2 = Norm(cfg.n_embd, **cfg.extra)
        self.mlp = MLP(cfg.n_embd, **cfg.extra)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class GPT(nn.Module):
    """A small pre-norm decoder-only transformer.

    Deliberately boring. The interesting parts live in components/ and are
    selected by name from the experiment config.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm_f = components.get("norm", cfg.norm)(cfg.n_embd, **cfg.extra)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init_weights)
        # Scale residual projections by depth, GPT-2 style.
        for name, p in self.named_parameters():
            if name.endswith(("attn.proj.weight", "mlp.w_down.weight")):
                nn.init.normal_(p, mean=0.0, std=0.02 / (2 * cfg.n_layer) ** 0.5)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()
            if not self.cfg.tie_embeddings:
                n -= self.lm_head.weight.numel()
        return n

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        x = self.tok_emb(idx)
        for block in self.blocks:
            x = block(x)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)).float(), targets.view(-1))
        return logits, loss
