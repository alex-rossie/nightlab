"""Helpers for the human review step: compare a result to its baseline and
render the PR comment. Used by the `lab` skill and the GPU workflow."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from . import ledger


def load_result(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def criterion_for(experiment: str) -> str:
    """The replication criterion paragraph from experiments/<name>/README.md, if any."""
    readme = ledger.ROOT / "experiments" / experiment / "README.md"
    if not readme.exists():
        return ""
    pattern = r"\*\*Replication criterion[^*]*\*\*\s*(.+?)(?:\n\n|\Z)"
    m = re.search(pattern, readme.read_text(), re.S)
    return m.group(1).strip() if m else ""


def compare(result: dict[str, Any]) -> dict[str, Any]:
    entries = ledger.load()
    base = ledger.baseline_for(entries, result["hardware"])
    out: dict[str, Any] = {
        "run_id": result["run_id"],
        "experiment": result["experiment"],
        "hardware": result["hardware"],
        "val_loss": result["val_loss"],
        "tokens_per_second": result["tokens_per_second"],
        "budget_ok": abs(result["wall_seconds"] - result["budget_seconds"]) < 5,
        "clean_commit": bool(re.fullmatch(r"[0-9a-f]{40}", result["commit"])),
        "criterion": criterion_for(result["experiment"]),
        "baseline": None,
    }
    if base is not None:
        out["baseline"] = {
            "run_id": base["run_id"],
            "val_loss": base["val_loss"],
            "tokens_per_second": base["tokens_per_second"],
            "finished_at": base["finished_at"][:10],
            "commit": base["commit"][:10],
        }
        out["delta_val_loss"] = round(result["val_loss"] - base["val_loss"], 4)
        out["delta_tokens_per_second_pct"] = round(
            100 * (result["tokens_per_second"] / base["tokens_per_second"] - 1), 1
        )
    return out


def format_compare(c: dict[str, Any]) -> str:
    prov = "clean commit" if c["clean_commit"] else "NOT LEDGER-ELIGIBLE (dirty/unknown commit)"
    budget = "budget ok" if c["budget_ok"] else "BUDGET MISMATCH"
    lines = [
        f"run        {c['run_id']}",
        f"experiment {c['experiment']}",
        f"hardware   {c['hardware']}",
        f"provenance {prov}, {budget}",
    ]
    if c["baseline"] is None:
        lines.append("baseline   none on this hardware; this run cannot be compared")
    else:
        b = c["baseline"]
        dl, dt = c["delta_val_loss"], c["delta_tokens_per_second_pct"]
        lines.append(f"baseline   {b['run_id']} ({b['finished_at']}, {b['commit']})")
        lines.append(f"val loss   {c['val_loss']:.4f} vs {b['val_loss']:.4f}  Δ {dl:+.4f}")
        tps, btps = c["tokens_per_second"], b["tokens_per_second"]
        lines.append(f"tokens/s   {tps:,.0f} vs {btps:,.0f}  Δ {dt:+.1f}%")
    crit = c["criterion"] or "(none written; the experiment README must state one)"
    lines.append("criterion  " + crit)
    return "\n".join(lines)


def pr_comment(result: dict[str, Any]) -> str:
    r = result
    mem = f"{r['peak_memory_gb']} GB" if r.get("peak_memory_gb") is not None else "n/a"
    lines = [
        f"**GPU run complete** · `{r['run_id']}`",
        "",
        "| val loss | tokens/s | steps | tokens | peak mem | wall | commit |",
        "|---:|---:|---:|---:|---:|---:|---|",
        f"| {r['val_loss']:.4f} | {r['tokens_per_second']:,.0f} | {r['steps']} | {r['tokens']:,} "
        f"| {mem} | {r['wall_seconds']} s | `{r['commit'][:10]}` |",
        "",
        "Curve: " + ", ".join(f"{c['t']:.0f}s→{c['val_loss']:.3f}" for c in r["curve"]),
    ]
    c = compare(r)
    if c["baseline"] is not None:
        dl, dt = c["delta_val_loss"], c["delta_tokens_per_second_pct"]
        lines += ["", f"Against baseline `{c['baseline']['run_id'][-6:]}`: "
                      f"Δ val loss {dl:+.4f}, Δ tokens/s {dt:+.1f}%."]
    if c["criterion"]:
        lines += ["", f"Criterion: {c['criterion']}"]
    lines += ["", "Reviewer: set the verdict, then "
                  "`uv run nightlab ledger-add result.json --verdict <v>`."]
    return "\n".join(lines)
