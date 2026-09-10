"""Experiment configuration.

Every experiment is a YAML file. The schema is deliberately small: a model, an
optimizer, a training budget, and a data source. Anything an experiment wants
to vary must be expressible here or as a registered component, so that two
runs are always comparable field by field.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    block_size: int = 1024
    vocab_size: int = 50304  # GPT-2 vocab padded to a multiple of 64
    attention: str = "causal"
    mlp: str = "swiglu"
    norm: str = "rmsnorm"
    tie_embeddings: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimConfig:
    name: str = "adamw"
    lr: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    warmup_steps: int = 100
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainConfig:
    batch_size: int = 16
    seq_len: int = 1024
    wall_seconds: int = 600
    eval_every_seconds: int = 60
    eval_batches: int = 20
    grad_clip: float = 1.0
    compile: bool = True
    dtype: str = "bfloat16"


@dataclass
class DataConfig:
    dataset: str = "fineweb-edu"
    path: str = "data"


@dataclass
class ExperimentConfig:
    name: str = "baseline"
    claim: str = ""
    source: str = ""
    seed: int = 1337
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        raw = yaml.safe_load(Path(path).read_text()) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ExperimentConfig:
        cfg = cls()
        for key, value in raw.items():
            if key in ("model", "optim", "train", "data"):
                section = getattr(cfg, key)
                for k, v in (value or {}).items():
                    if not hasattr(section, k):
                        raise KeyError(f"unknown {key} option: {k}")
                    if k == "betas":
                        v = tuple(v)
                    setattr(section, k, v)
            elif hasattr(cfg, key):
                setattr(cfg, key, value)
            else:
                raise KeyError(f"unknown top-level option: {key}")
        return cfg

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["optim"]["betas"] = list(d["optim"]["betas"])
        return d
