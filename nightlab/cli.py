from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import ledger
from .config import ExperimentConfig


def smoke_config(cfg: ExperimentConfig) -> ExperimentConfig:
    """Keep the experiment's component choices, shrink everything else to CPU size."""
    cfg.model.n_layer, cfg.model.n_head, cfg.model.n_embd, cfg.model.block_size = 2, 2, 64, 128
    cfg.optim.warmup_steps = 5
    cfg.train.batch_size, cfg.train.seq_len = 4, 128
    cfg.train.wall_seconds, cfg.train.eval_every_seconds, cfg.train.eval_batches = 20, 10, 2
    cfg.train.compile, cfg.train.dtype = False, "float32"
    cfg.data.dataset = "synthetic"
    return cfg


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="nightlab")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="run one fixed-budget experiment")
    t.add_argument("config", help="path to experiment YAML")
    t.add_argument("--out", help="run directory (default runs/<run_id>)")
    t.add_argument("--wall-seconds", type=int, help="override the budget (smoke tests only)")
    t.add_argument("--dataset", choices=["synthetic", "tinyshakespeare", "fineweb-edu"],
                   help="override the dataset (smoke tests only)")
    t.add_argument("--smoke", action="store_true",
                   help="shrink to a 2-layer model, synthetic data, 20 s, float32, no compile")
    t.add_argument("--quiet", action="store_true")

    d = sub.add_parser("data", help="download and tokenize a dataset")
    d.add_argument("dataset", choices=["synthetic", "tinyshakespeare", "fineweb-edu"])
    d.add_argument("--max-tokens", type=int)
    d.add_argument("--root", default="data")

    la = sub.add_parser("ledger-add", help="append a result.json to the ledger")
    la.add_argument("result", help="path to result.json")
    la.add_argument("--verdict", choices=ledger.VERDICTS, default="pending")
    la.add_argument("--notes", default="")

    sub.add_parser("ledger-validate", help="check the ledger for schema or provenance problems")

    r = sub.add_parser("render", help="regenerate README tables from the ledger")
    r.add_argument("--check", action="store_true", help="fail if README is stale")

    args = p.parse_args(argv)

    if args.cmd == "train":
        from .train import run

        cfg = ExperimentConfig.from_yaml(args.config)
        if args.smoke:
            cfg = smoke_config(cfg)
        if args.wall_seconds:
            cfg.train.wall_seconds = args.wall_seconds
        if args.dataset:
            cfg.data.dataset = args.dataset
        result = run(cfg, out_dir=args.out, quiet=args.quiet)
        print(f"val_loss={result['val_loss']} tokens/s={result['tokens_per_second']} "
              f"steps={result['steps']} run_id={result['run_id']}")
        return 0

    if args.cmd == "data":
        from .data import prepare

        out = prepare(args.dataset, args.root, args.max_tokens)
        print(out)
        return 0

    if args.cmd == "ledger-add":
        entry = json.loads(Path(args.result).read_text())
        entry["verdict"] = args.verdict
        if args.notes:
            entry["notes"] = args.notes
        ledger.append(entry)
        print(f"added {entry['run_id']} as {args.verdict}")
        return 0

    if args.cmd == "ledger-validate":
        problems = ledger.validate_file()
        for pr in problems:
            print(pr, file=sys.stderr)
        print(f"{len(ledger.load())} entries, {len(problems)} problems")
        return 1 if problems else 0

    if args.cmd == "render":
        from .render import render_readme

        changed = render_readme(check=args.check)
        if args.check and changed:
            print("README.md is stale; run `uv run nightlab render`", file=sys.stderr)
            return 1
        print("README.md up to date" if not changed else "README.md updated")
        return 0

    return 2


if __name__ == "__main__":
    sys.exit(main())
