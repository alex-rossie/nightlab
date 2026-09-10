"""Datasets.

Everything is tokenized once with the GPT-2 tokenizer into flat uint16 files
so that training is a random slice of a memory-mapped array. Three sources:

  synthetic      random tokens, for CI and smoke tests
  tinyshakespeare ~300k tokens, for local sanity checks
  fineweb-edu    the real thing, a slice of the 10BT sample (needs `datasets`)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import requests

TINY_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
EOT = 50256


def _enc():
    import tiktoken

    return tiktoken.get_encoding("gpt2")


def _write(tokens: np.ndarray, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tokens.astype(np.uint16).tofile(out)


def prepare(dataset: str, root: str | Path = "data", max_tokens: int | None = None) -> Path:
    """Materialize <root>/<dataset>/{train,val}.bin and return the directory."""
    out_dir = Path(root) / dataset
    if (out_dir / "train.bin").exists() and (out_dir / "val.bin").exists():
        return out_dir

    if dataset == "synthetic":
        rng = np.random.default_rng(0)
        n = max_tokens or 2_000_000
        train = rng.integers(0, 50257, size=n, dtype=np.uint16)
        val = rng.integers(0, 50257, size=n // 10, dtype=np.uint16)
    elif dataset == "tinyshakespeare":
        text = requests.get(TINY_URL, timeout=60).text
        ids = np.array(_enc().encode_ordinary(text), dtype=np.uint16)
        split = int(0.9 * len(ids))
        train, val = ids[:split], ids[split:]
    elif dataset == "fineweb-edu":
        train, val = _fineweb_edu(max_tokens or 300_000_000)
    else:
        raise ValueError(f"unknown dataset {dataset!r}")

    _write(train, out_dir / "train.bin")
    _write(val, out_dir / "val.bin")
    return out_dir


def _fineweb_edu(max_tokens: int) -> tuple[np.ndarray, np.ndarray]:
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("fineweb-edu needs the data extra: uv sync --extra data") from None

    enc = _enc()
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True
    )
    chunks: list[np.ndarray] = []
    total = 0
    for row in ds:
        ids = enc.encode_ordinary(row["text"]) + [EOT]
        chunks.append(np.array(ids, dtype=np.uint16))
        total += len(ids)
        if total >= max_tokens:
            break
    tokens = np.concatenate(chunks)
    # Fixed 10M-token validation slice, taken from the front so it never moves
    # when max_tokens changes.
    n_val = min(10_000_000, len(tokens) // 20)
    return tokens[n_val:], tokens[:n_val]


class TokenSampler:
    """Random contiguous windows from a memory-mapped token file."""

    def __init__(self, path: Path, seq_len: int, batch_size: int, seed: int):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)

    def __call__(self):
        import torch

        n = len(self.data) - self.seq_len - 1
        starts = self.rng.integers(0, n, size=self.batch_size)
        x = np.stack([self.data[s : s + self.seq_len] for s in starts]).astype(np.int64)
        y = np.stack([self.data[s + 1 : s + 1 + self.seq_len] for s in starts]).astype(np.int64)
        return torch.from_numpy(x), torch.from_numpy(y)
