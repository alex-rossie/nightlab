"""The claims ledger.

One JSONL file, one line per completed run. It is the only source of truth for
anything the README says about results. Entries are validated against
ledger/schema.json for shape and against `provenance_problems` for the
promises the prose makes: exactly the stated budget, the real dataset, a GPU,
token accounting that adds up, a comparable baseline, and a second seed behind
every `replicated`. Verdicts are assigned by a human reviewer on the PR, never
by the agent that ran the experiment.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import jsonschema

ROOT = Path(__file__).resolve().parent.parent
LEDGER_PATH = ROOT / "ledger" / "results.jsonl"
SCHEMA_PATH = ROOT / "ledger" / "schema.json"

VERDICTS = ("baseline", "replicated", "partial", "failed", "inconclusive", "pending")
BUDGET_SECONDS = 600
DATASET = "fineweb-edu"
# Two runs are comparable when these agree, along with the whole train section
# of the config and the dataset. Together they pin the evaluation protocol and
# the exact tokenized data, so a baseline scored under an older loop is never
# used for a delta.
COMPARABLE_FIELDS = ("hardware", "final_eval_batches", "data_fingerprint")


def schema() -> dict[str, Any]:
    return json.loads(SCHEMA_PATH.read_text())


def validate(entry: dict[str, Any]) -> None:
    """Shape only. Provenance is `provenance_problems`."""
    jsonschema.validate(entry, schema(), format_checker=jsonschema.FormatChecker())


def load(path: Path = LEDGER_PATH) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    entries = []
    for i, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}:{i}: bad JSON: {e}") from None
    return entries


def _config(entry: dict[str, Any]) -> dict[str, Any]:
    return entry.get("config") or {}


def comparability_problems(entry: dict[str, Any], base: dict[str, Any]) -> list[str]:
    """Why `entry` must not be scored against `base`. Empty means it may."""
    problems = []
    for f in COMPARABLE_FIELDS:
        if entry.get(f) != base.get(f):
            problems.append(f"{f} {entry.get(f)!r} vs baseline {base.get(f)!r}")
    if _config(entry).get("train") != _config(base).get("train"):
        problems.append("config.train differs from the baseline's")
    mine = _config(entry).get("data", {}).get("dataset")
    theirs = _config(base).get("data", {}).get("dataset")
    if mine != theirs:
        problems.append(f"dataset {mine!r} vs baseline {theirs!r}")
    return problems


def baseline_for(
    entries: list[dict[str, Any]], hardware: str, like: dict[str, Any] | None = None
) -> dict[str, Any] | None:
    """Most recent baseline run on the same hardware. With `like`, the most recent
    one that `like` may be compared against, or None if there is none."""
    candidates = [e for e in entries if e["verdict"] == "baseline" and e["hardware"] == hardware]
    if like is not None:
        candidates = [e for e in candidates if not comparability_problems(like, e)]
    return max(candidates, key=lambda e: e["finished_at"]) if candidates else None


def _parse_time(s: str) -> datetime | None:
    try:
        return datetime.fromisoformat(s)
    except (TypeError, ValueError):
        return None


def provenance_problems(entry: dict[str, Any], entries: list[dict[str, Any]]) -> list[str]:
    """Why this entry should not count, as a list of reasons. Empty means it may.
    `entries` is the whole ledger the entry is judged with, itself included."""
    problems = []
    cfg = _config(entry)
    train, data = cfg.get("train", {}), cfg.get("data", {})
    if entry["budget_seconds"] != BUDGET_SECONDS:
        problems.append(f"budget is {entry['budget_seconds']} s, not {BUDGET_SECONDS}")
    if train.get("wall_seconds") != entry["budget_seconds"]:
        problems.append("config.train.wall_seconds disagrees with budget_seconds")
    if abs(entry["wall_seconds"] - entry["budget_seconds"]) >= 5:
        problems.append(f"wall_seconds {entry['wall_seconds']} is not the budget")
    if data.get("dataset") != DATASET:
        problems.append(f"dataset is {data.get('dataset')!r}, not {DATASET!r}")
    if entry["hardware"].startswith("cpu-") or entry.get("device", "cuda") != "cuda":
        problems.append("not a GPU run")
    if "final_eval_batches" in entry and "device" not in entry:
        problems.append("device is missing")
    expected = entry["steps"] * train.get("batch_size", 0) * train.get("seq_len", 0)
    if entry["tokens"] != expected:
        problems.append(f"tokens {entry['tokens']} != steps * batch_size * seq_len {expected}")
    if entry["seed"] != cfg.get("seed"):
        problems.append(f"seed {entry['seed']} disagrees with config.seed {cfg.get('seed')}")
    if entry["experiment"] != cfg.get("name"):
        problems.append(f"experiment {entry['experiment']!r} disagrees with config.name")
    started, finished = _parse_time(entry["started_at"]), _parse_time(entry["finished_at"])
    if started is None or finished is None:
        problems.append("started_at or finished_at is not an ISO 8601 timestamp")
    elif finished <= started:
        problems.append("finished_at is not after started_at")
    others = [e for e in entries if e["run_id"] != entry["run_id"]]
    if entry["verdict"] != "baseline":
        if "pr" not in entry:
            problems.append("pr is missing; every verdict names the PR it was reviewed on")
        base_id = entry.get("baseline_run_id")
        base = next((e for e in others if e["run_id"] == base_id), None)
        if base is None:
            problems.append("baseline_run_id is missing or not in the ledger")
        elif base["verdict"] != "baseline":
            problems.append(f"baseline_run_id {base_id} is not a baseline")
        else:
            problems += [f"not comparable to baseline {base_id}: {p}"
                         for p in comparability_problems(entry, base)]
    if entry["verdict"] == "replicated":
        def sans_seed(c: dict[str, Any]) -> dict[str, Any]:
            return {k: v for k, v in c.items() if k != "seed"}

        second = [e for e in others
                  if e["experiment"] == entry["experiment"] and e["hardware"] == entry["hardware"]
                  and e["seed"] != entry["seed"] and e["verdict"] in ("replicated", "partial")
                  and sans_seed(_config(e)) == sans_seed(cfg)]
        if not second:
            problems.append("replicated needs a second seed of the same config on the same "
                            "hardware, judged replicated or partial, in the ledger")
    return problems


def append_all(new: list[dict[str, Any]], path: Path = LEDGER_PATH) -> None:
    """Append several entries as one unit, judged together. Two seeds of one
    experiment can enter as `replicated` in the same call; alone, neither could."""
    for e in new:
        validate(e)
    existing = load(path)
    ids = {e["run_id"] for e in existing}
    problems = []
    for e in new:
        if e["run_id"] in ids:
            problems.append(f"{e['run_id']}: already in ledger")
        ids.add(e["run_id"])
    everything = existing + list(new)
    for e in new:
        problems += [f"{e['run_id']}: {p}" for p in provenance_problems(e, everything)]
    if problems:
        raise ValueError("; ".join(problems))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for e in new:
            f.write(json.dumps(e, sort_keys=True) + "\n")


def append(entry: dict[str, Any], path: Path = LEDGER_PATH) -> None:
    append_all([entry], path)


def validate_file(path: Path = LEDGER_PATH) -> list[str]:
    """Return a list of problems; empty means the ledger is clean."""
    problems = []
    seen = set()
    entries = load(path)
    for i, entry in enumerate(entries, 1):
        try:
            validate(entry)
        except jsonschema.ValidationError as e:
            problems.append(f"line {i}: {e.message}")
            continue
        if entry["run_id"] in seen:
            problems.append(f"line {i}: duplicate run_id {entry['run_id']}")
        seen.add(entry["run_id"])
        problems += [f"line {i}: {p}" for p in provenance_problems(entry, entries)]
    return problems
