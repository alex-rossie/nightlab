"""Fixed-budget training.

The contract: train for exactly `train.wall_seconds` of wall-clock time on the
stated hardware, then report validation loss. Nothing else is held constant
across experiments on purpose. A change that trains slower per step but reaches
a lower loss in the same ten minutes is a real improvement. A change that only
wins per step is not, and this loop is designed to expose that.
"""

from __future__ import annotations

import json
import math
import os
import platform
import subprocess
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from . import __version__, components
from .config import ExperimentConfig
from .data import TokenSampler, prepare
from .model import GPT


def git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        commit = out.decode().strip()
        dirty = subprocess.call(["git", "diff", "--quiet"], stderr=subprocess.DEVNULL) != 0
        return commit + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def hardware_id() -> str:
    """Ledger hardware string. NIGHTLAB_HARDWARE overrides the device name, so a lab
    can publish results under a class label rather than an exact card."""
    if os.environ.get("NIGHTLAB_HARDWARE"):
        return os.environ["NIGHTLAB_HARDWARE"]
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        return name.lower().replace(" ", "-").replace("/", "-")
    return f"cpu-{platform.machine()}"


def lr_at(step: int, warmup: int, base_lr: float) -> float:
    # Warmup then constant. No decay, because the budget is wall time and we
    # do not know the final step count in advance. Experiments may register
    # schedulers later; keep the baseline honest and simple.
    return base_lr * min(1.0, (step + 1) / max(1, warmup))


@torch.no_grad()
def evaluate(model, sampler, n_batches: int, device: str, autocast) -> float:
    model.eval()
    losses = []
    for _ in range(n_batches):
        x, y = sampler()
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with autocast():
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(sum(losses) / len(losses))


def run(cfg: ExperimentConfig, out_dir: Path | None = None, quiet: bool = False) -> dict[str, Any]:
    torch.manual_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[
        cfg.train.dtype
    ]

    def autocast():
        return torch.autocast(device_type=device, dtype=dtype, enabled=dtype != torch.float32)

    data_dir = prepare(cfg.data.dataset, cfg.data.path)
    seq, bs = cfg.train.seq_len, cfg.train.batch_size
    train_sampler = TokenSampler(data_dir / "train.bin", seq, bs, cfg.seed)
    val_sampler = TokenSampler(data_dir / "val.bin", seq, bs, seed=0)

    model = GPT(cfg.model).to(device)
    n_params = model.num_params()
    optimizers = components.get("optim", cfg.optim.name)(model, cfg.optim)
    base_lrs = [[g["lr"] for g in opt.param_groups] for opt in optimizers]
    compiled = torch.compile(model) if (cfg.train.compile and device == "cuda") else model

    run_id = f"{cfg.name}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:6]}"
    out_dir = Path(out_dir or f"runs/{run_id}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    log = (out_dir / "log.jsonl").open("w")

    def emit(rec: dict[str, Any]) -> None:
        log.write(json.dumps(rec) + "\n")
        log.flush()
        if not quiet:
            print(json.dumps(rec), flush=True)

    emit({"event": "start", "run_id": run_id, "params": n_params, "device": device,
          "hardware": hardware_id(), "commit": git_commit()})

    started_at = datetime.now(UTC)
    t0 = time.time()
    next_eval = t0 + cfg.train.eval_every_seconds
    step, tokens = 0, 0
    tokens_per_step = cfg.train.batch_size * cfg.train.seq_len
    curve: list[dict[str, float]] = []
    train_loss = float("nan")

    while True:
        elapsed = time.time() - t0
        if elapsed >= cfg.train.wall_seconds:
            break
        if time.time() >= next_eval:
            vl = evaluate(compiled, val_sampler, cfg.train.eval_batches, device, autocast)
            curve.append({"t": round(elapsed, 1), "step": step, "tokens": tokens, "val_loss": vl})
            emit({"event": "eval", **curve[-1]})
            next_eval += cfg.train.eval_every_seconds

        for opt, lrs in zip(optimizers, base_lrs, strict=True):
            for g, base in zip(opt.param_groups, lrs, strict=True):
                g["lr"] = lr_at(step, cfg.optim.warmup_steps, base)

        x, y = train_sampler()
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with autocast():
            _, loss = compiled(x, y)
        loss.backward()
        if cfg.train.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        for opt in optimizers:
            opt.step()
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        step += 1
        tokens += tokens_per_step
        train_loss = loss.item()
        if step % 20 == 0:
            emit({"event": "step", "step": step, "t": round(time.time() - t0, 1),
                  "train_loss": round(train_loss, 4)})

    wall = time.time() - t0
    if device == "cuda":
        torch.cuda.synchronize()
    final_val = evaluate(compiled, val_sampler, cfg.train.eval_batches * 2, device, autocast)
    curve.append({"t": round(wall, 1), "step": step, "tokens": tokens, "val_loss": final_val})
    peak_mem = torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else None

    result = {
        "run_id": run_id,
        "experiment": cfg.name,
        "claim": cfg.claim,
        "source": cfg.source,
        "verdict": "pending",
        "commit": git_commit(),
        "nightlab_version": __version__,
        "seed": cfg.seed,
        "hardware": hardware_id(),
        "torch": torch.__version__,
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
        "wall_seconds": round(wall, 1),
        "budget_seconds": cfg.train.wall_seconds,
        "params": n_params,
        "steps": step,
        "tokens": tokens,
        "tokens_per_second": round(tokens / wall, 1),
        "peak_memory_gb": round(peak_mem, 2) if peak_mem is not None else None,
        "final_train_loss": round(train_loss, 4) if not math.isnan(train_loss) else None,
        "val_loss": round(final_val, 4),
        "curve": curve,
        "config": cfg.to_dict(),
        "notes": "",
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    emit({"event": "done", "run_id": run_id, "val_loss": result["val_loss"],
          "tokens_per_second": result["tokens_per_second"], "steps": step})
    log.close()
    return result


if __name__ == "__main__":
    import sys

    cfg = ExperimentConfig.from_yaml(sys.argv[1])
    os.chdir(Path(__file__).resolve().parent.parent)
    run(cfg)
