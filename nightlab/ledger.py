"""The claims ledger.

One JSONL file, one line per completed run. It is the only source of truth for
anything the README says about results. Entries are validated against
ledger/schema.json and rejected if they are missing provenance (commit, seed,
hardware, wall time). Verdicts are assigned by a human reviewer on the PR,
never by the agent that ran the experiment.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema

ROOT = Path(__file__).resolve().parent.parent
LEDGER_PATH = ROOT / "ledger" / "results.jsonl"
SCHEMA_PATH = ROOT / "ledger" / "schema.json"

VERDICTS = ("baseline", "replicated", "partial", "failed", "inconclusive", "pending")


def schema() -> dict[str, Any]:
    return json.loads(SCHEMA_PATH.read_text())


def validate(entry: dict[str, Any]) -> None:
    jsonschema.validate(entry, schema())


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


def append(entry: dict[str, Any], path: Path = LEDGER_PATH) -> None:
    validate(entry)
    existing = {e["run_id"] for e in load(path)}
    if entry["run_id"] in existing:
        raise ValueError(f"run_id {entry['run_id']} already in ledger")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(entry, sort_keys=True) + "\n")


def validate_file(path: Path = LEDGER_PATH) -> list[str]:
    """Return a list of problems; empty means the ledger is clean."""
    problems = []
    seen = set()
    for i, entry in enumerate(load(path), 1):
        try:
            validate(entry)
        except jsonschema.ValidationError as e:
            problems.append(f"line {i}: {e.message}")
            continue
        if entry["run_id"] in seen:
            problems.append(f"line {i}: duplicate run_id {entry['run_id']}")
        seen.add(entry["run_id"])
    return problems


def baseline_for(entries: list[dict[str, Any]], hardware: str) -> dict[str, Any] | None:
    """Most recent baseline run on the same hardware."""
    candidates = [e for e in entries if e["verdict"] == "baseline" and e["hardware"] == hardware]
    return max(candidates, key=lambda e: e["finished_at"]) if candidates else None
