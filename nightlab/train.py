"""Fixed-budget training.

The contract: train for exactly `train.wall_seconds` of wall-clock time on the
stated hardware, then report validation loss. Nothing else is held constant
across experiments on purpose. A change that trains slower per step but reaches
a lower loss in the same ten minutes is a real improvement. A change that only
wins per step is not, and this loop is designed to expose that.

Two things are held constant deliberately, because they are not the technique:

* Compile state. Every run points the on-disk compiler caches at an empty
  directory, so every run pays the same cold compile inside the budget. A warm
  cache left by an earlier run with the same graph would otherwise hand that
  run 15-20 s of extra training. One run per process; the in-process caches
  are not reset between runs.
* The validation set. Evaluation reads fixed, non-overlapping windows from the
  front of val.bin, in order. The final number is the whole slice; the periodic
  curve points are its first `eval_batches` batches. The result records the
  batch count and a fingerprint of the data so the ledger can refuse to
  compare runs scored differently.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import platform
import shutil
import subprocess
import tempfile
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from . import __version__, components
from .config import ExperimentConfig
from .data import FixedWindows, TokenSampler, fingerprint, prepare
from .model import GPT


def git_commit() -> str:
    """HEAD, suffixed -dirty if anything is modified, staged, or untracked.

    Untracked files count. A new component that was never `git add`ed still
    changes what ran, and the ledger must not carry a hash that lacks it.
    """
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        commit = out.decode().strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        )
        return commit + ("-dirty" if status.strip() else "")
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


def device_name(device: str) -> str:
    """The physical device, recorded next to the hardware label so the two can be
    checked against each other."""
    if device == "cuda":
        return torch.cuda.get_device_name(0)
    return platform.processor() or platform.machine()


class FreshCompileCache:
    """Point inductor and Triton at an empty directory for the duration of a run,
    then remove it and put the environment back."""

    VARS = ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR")

    def __enter__(self) -> Path:
        import torch._dynamo

        self.saved = {k: os.environ.get(k) for k in self.VARS}
        self.path = Path(tempfile.mkdtemp(prefix="nightlab-inductor-"))
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(self.path)
        os.environ["TRITON_CACHE_DIR"] = str(self.path / "triton")
        torch._dynamo.reset()
        return self.path

    def __exit__(self, *exc) -> bool:
        shutil.rmtree(self.path, ignore_errors=True)
        for k, v in self.saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        return False


def lr_at(step: int, warmup: int, base_lr: float) -> float:
    # Warmup then constant. No decay, because the budget is wall time and we
    # do not know the final step count in advance. Experiments may register
    # schedulers later; keep the baseline honest and simple.
    return base_lr * min(1.0, (step + 1) / max(1, warmup))


@torch.no_grad()
def evaluate(model, batches, device: str, autocast) -> float:
    model.eval()
    total, n = 0.0, 0
    for x, y in batches:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with autocast():
            _, loss = model(x, y)
        total += loss.item()
        n += 1
    model.train()
    if n == 0:
        raise ValueError("evaluate() got no batches")
    return total / n


def run(
    cfg: ExperimentConfig,
    out_dir: Path | None = None,
    quiet: bool = False,
    data_root: str | Path | None = None,
) -> dict[str, Any]:
    run_id = f"{cfg.name}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:6]}"
    out_dir = Path(out_dir or f"runs/{run_id}")
    for stale in ("result.json", "failure.json"):
        if (out_dir / stale).exists():
            raise FileExistsError(f"{out_dir} already holds {stale}; use another --out")
    if cfg.train.eval_batches < 1 or cfg.train.final_eval_batches < 0:
        raise ValueError("eval_batches must be >= 1 and final_eval_batches >= 0")

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

    data_root = data_root or os.environ.get("NIGHTLAB_DATA") or cfg.data.path
    data_dir = prepare(cfg.data.dataset, data_root)
    seq, bs = cfg.train.seq_len, cfg.train.batch_size
    train_sampler = TokenSampler(data_dir / "train.bin", seq, bs, cfg.seed)
    val = FixedWindows(data_dir / "val.bin", seq, bs)
    n_final = min(cfg.train.final_eval_batches, len(val)) or len(val)
    data_tokens = {"train": train_sampler.n_tokens, "val": val.n_tokens}
    data_fp = fingerprint(data_dir / "train.bin", data_dir / "val.bin")

    model = GPT(cfg.model).to(device)
    n_params = model.num_params()
    optimizers = components.get("optim", cfg.optim.name)(model, cfg.optim)
    base_lrs = [[g["lr"] for g in opt.param_groups] for opt in optimizers]
    use_compile = cfg.train.compile and device == "cuda"
    cache = FreshCompileCache() if use_compile else contextlib.nullcontext()

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    commit = git_commit()
    started_at = datetime.now(UTC)

    with cache, (out_dir / "log.jsonl").open("w") as log:

        def emit(rec: dict[str, Any]) -> None:
            log.write(json.dumps(rec) + "\n")
            log.flush()
            if not quiet:
                print(json.dumps(rec), flush=True)

        compiled = torch.compile(model) if use_compile else model
        emit({"event": "start", "run_id": run_id, "params": n_params, "device": device,
              "device_name": device_name(device), "hardware": hardware_id(), "commit": commit,
              "data_dir": str(data_dir), "data_tokens": data_tokens,
              "data_fingerprint": data_fp})

        clock = time.perf_counter
        t0 = clock()
        next_eval = t0 + cfg.train.eval_every_seconds
        step, tokens = 0, 0
        tokens_per_step = cfg.train.batch_size * cfg.train.seq_len
        curve: list[dict[str, float]] = []
        train_loss = float("nan")
        compile_seconds: float | None = None
        first_eval_seconds: float | None = None

        try:
            while True:
                now = clock()
                if now - t0 >= cfg.train.wall_seconds:
                    break
                if now >= next_eval:
                    t_eval = clock()
                    vl = evaluate(compiled, val.batches(cfg.train.eval_batches), device,
                                  autocast)
                    if first_eval_seconds is None:
                        first_eval_seconds = round(clock() - t_eval, 1)
                    curve.append({"t": round(now - t0, 1), "step": step, "tokens": tokens,
                                  "val_loss": vl})
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
                if step == 1:
                    compile_seconds = round(clock() - t0, 1)
                if step % 20 == 0:
                    emit({"event": "step", "step": step, "t": round(clock() - t0, 1),
                          "train_loss": round(train_loss, 4)})

            wall = clock() - t0
            if device == "cuda":
                torch.cuda.synchronize()
            final_val = evaluate(compiled, val.batches(n_final), device, autocast)
            curve.append({"t": round(wall, 1), "step": step, "tokens": tokens,
                          "val_loss": final_val})
        except Exception as e:
            emit({"event": "failed", "run_id": run_id, "error": repr(e), "step": step})
            failure = {"run_id": run_id, "error": repr(e), "step": step, "tokens": tokens,
                       "curve": curve, "commit": commit, "hardware": hardware_id(),
                       "config": cfg.to_dict()}
            with contextlib.suppress(OSError):
                (out_dir / "failure.json").write_text(json.dumps(failure, indent=2))
            raise

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
            "device": device,
            "device_name": device_name(device),
            "torch": torch.__version__,
            "started_at": started_at.isoformat(),
            "finished_at": datetime.now(UTC).isoformat(),
            "wall_seconds": round(wall, 1),
            "budget_seconds": cfg.train.wall_seconds,
            "compile_seconds": compile_seconds,
            "first_eval_seconds": first_eval_seconds,
            "params": n_params,
            "steps": step,
            "tokens": tokens,
            "tokens_per_second": round(tokens / wall, 1),
            "peak_memory_gb": round(peak_mem, 2) if peak_mem is not None else None,
            "final_train_loss": round(train_loss, 4) if not math.isnan(train_loss) else None,
            "val_loss": round(final_val, 4),
            "final_eval_batches": n_final,
            "data_tokens": data_tokens,
            "data_fingerprint": data_fp,
            "curve": curve,
            "config": cfg.to_dict(),
            "notes": "",
        }
        (out_dir / "result.json").write_text(json.dumps(result, indent=2))
        emit({"event": "done", "run_id": run_id, "val_loss": result["val_loss"],
              "tokens_per_second": result["tokens_per_second"], "steps": step})
    return result


if __name__ == "__main__":
    import sys

    cfg = ExperimentConfig.from_yaml(sys.argv[1])
    os.chdir(Path(__file__).resolve().parent.parent)
    run(cfg)
