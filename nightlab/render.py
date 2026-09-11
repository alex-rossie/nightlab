"""Render the scoreboard and results table into README.md from the ledger.

Everything between the ledger markers is machine-owned. Edit the ledger, not
the README.
"""

from __future__ import annotations

import re
from collections import Counter

from . import ledger

README = ledger.ROOT / "README.md"
START, END = "<!-- ledger:start -->", "<!-- ledger:end -->"

VERDICT_ICON = {
    "baseline": "⚪",
    "replicated": "🟢",
    "partial": "🟡",
    "failed": "🔴",
    "inconclusive": "⚫",
    "pending": "⏳",
}


def _fmt_delta(entry, base) -> str:
    if base is None or entry["verdict"] == "baseline":
        return "—"
    d = entry["val_loss"] - base["val_loss"]
    return f"{d:+.3f}"


def build_section(entries: list[dict]) -> str:
    if not entries:
        return (
            "**Night 0.** Nothing has run on the reference hardware yet. "
            "The baseline goes first, then the queue in `experiments/`.\n"
        )
    counts = Counter(e["verdict"] for e in entries)
    tested = sum(counts[v] for v in ("replicated", "partial", "failed", "inconclusive"))
    nights = len({e["finished_at"][:10] for e in entries})
    lines = [
        "| Runs judged | Replicated | Partial | Failed | Inconclusive | Nights active |",
        "|---:|---:|---:|---:|---:|---:|",
        f"| {tested} | {counts['replicated']} | {counts['partial']} | {counts['failed']} "
        f"| {counts['inconclusive']} | {nights} |",
        "",
        "| | Experiment | Claim | Val loss | Δ vs baseline | Tok/s | Date | Run |",
        "|---|---|---|---:|---:|---:|---|---|",
    ]
    by_id = {e["run_id"]: e for e in entries}
    for e in sorted(entries, key=lambda e: e["finished_at"], reverse=True):
        # The baseline a verdict was judged against is pinned at ledger-add time;
        # older rows without one fall back to the newest comparable baseline.
        base = by_id.get(e.get("baseline_run_id", "")) or ledger.baseline_for(
            entries, e["hardware"], like=e
        )
        src = f" ([src]({e['source']}))" if e.get("source", "").startswith("http") else ""
        claim = (e.get("claim") or "").replace("|", "\\|")
        lines.append(
            f"| {VERDICT_ICON.get(e['verdict'], '?')} | `{e['experiment']}` "
            f"| {claim}{src} | {e['val_loss']:.3f} | {_fmt_delta(e, base)} "
            f"| {e['tokens_per_second']:,.0f} "
            f"| {e['finished_at'][:10]} | `{e['run_id'][-6:]}` |"
        )
    lines.append("")
    lines.append(
        "🟢 replicated · 🟡 partial · 🔴 failed · ⚫ inconclusive · ⚪ baseline "
        "· ⏳ awaiting review. Lower val loss is better. Every row is one run of exactly "
        "the stated budget on the stated hardware; full provenance is in "
        "`ledger/results.jsonl`."
    )
    return "\n".join(lines) + "\n"


def render_readme(check: bool = False) -> bool:
    text = README.read_text()
    if START not in text or END not in text:
        raise RuntimeError("README.md is missing ledger markers")
    section = build_section(ledger.load())
    new = re.sub(
        re.escape(START) + r".*?" + re.escape(END),
        lambda _: f"{START}\n{section}{END}",
        text,
        flags=re.S,
    )
    changed = new != text
    if changed and not check:
        README.write_text(new)
    return changed
