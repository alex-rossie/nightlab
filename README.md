# nightlab

**An autonomous research lab that lives in this repo.** Every night an agent reads the new LLM papers, picks one, reimplements it at small scale, trains it for exactly ten minutes on one consumer AMD GPU under ROCm, and publishes whether it replicated. A human reviews every result before it counts. Failures are published with the same care as successes.

The nightly loop is the product. The replication results are the exhaust.

## Scoreboard

<!-- ledger:start -->
| Papers tested | Replicated | Partial | Failed | Inconclusive | Nights active |
|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 | 0 | 1 |

| | Experiment | Claim | Val loss | Δ vs baseline | Tok/s | Date | Run |
|---|---|---|---:|---:|---:|---|---|
| ⚪ | `baseline` | Reference run. A pre-norm decoder with RoPE, RMSNorm, SwiGLU, untied embeddings, AdamW. | 4.303 | — | 135,084 | 2026-09-10 | `5f6e26` |

🟢 replicated · 🟡 partial · 🔴 failed · ⚫ inconclusive · ⚪ baseline · ⏳ awaiting review. Lower val loss is better. Every row is one run of exactly the stated budget on the stated hardware; full provenance is in `ledger/results.jsonl`.
<!-- ledger:end -->

## The rule

Every experiment gets **600 seconds of wall clock on one consumer AMD GPU running ROCm**, compile time included, against a fixed 300M-token slice of FineWeb-Edu. Not a step count, not a token count. Seconds. A technique that halves loss per step but doubles step time has not helped anyone, and this budget is designed to expose exactly that. Throughput is reported next to loss on every row. [Why this budget](docs/budget.md).

## How a night works

```
02:00  radar         scans arXiv, HF trending, and a short list of lab blogs
                     scores candidates on a 4-gate rubric, opens ≤3 issues
03:00  implement     takes the top queued issue, writes one registered component
                     or one config diff, smoke tests on CPU, opens a draft PR
       gpu runner    a human applies the needs-gpu label; the lab machine
                     runs the full budget and comments the result on the PR
05:00  housekeeping  bumps deps, checks links, flags stale results for re-run
06:00  ledger-sync   re-renders this README from ledger/results.jsonl
Sun    digest        a weekly writeup, failures first; a human edits and merges
Sun    audit         an adversarial pass over the week's merges looking for slop
```

Times are Pacific. Each routine is a checked-in prompt in [`.claude/routines/`](.claude/routines/), run as a scheduled Claude Code cloud routine. The routines can open issues and draft PRs. They cannot set a verdict, edit the ledger, or merge anything but dependency bumps. Every night each routine writes a report to [`reports/nights/`](reports/nights/), including the nights it got stuck.

## What counts

- **The ledger is the truth.** [`ledger/results.jsonl`](ledger/results.jsonl) holds one line per run with commit hash, seed, hardware, torch version, wall time, throughput, and the full loss curve. The scoreboard above is rendered from it and CI fails if they disagree.
- **Criterion before result.** Every experiment README states a numeric replication criterion before the GPU run. See [`experiments/0001-muon`](experiments/0001-muon/README.md) for the shape.
- **Humans set verdicts.** `replicated`, `partial`, `failed`, or `inconclusive`. The agent proposes. A person decides, on the PR, with the runner's numbers in front of them.
- **One change per experiment.** A new component in [`nightlab/components/`](nightlab/components/) or a config diff against [`experiments/baseline`](experiments/baseline/). Baseline components are never edited in place.

## What is in scope

Anything a 25M-parameter decoder trained for ten minutes can meaningfully exercise: attention variants, MLP and norm changes, optimizers, initialization, multi-token prediction, parallel decoding objectives, data ordering. Out of scope, and listed as such in the weekly digest rather than quietly skipped: claims that only appear above a billion parameters, long-context behavior, RL, and anything needing instruction data.

## Run it yourself

```
git clone https://github.com/alex-rossie/nightlab && cd nightlab
uv sync --extra cpu --extra dev
uv run nightlab train experiments/baseline/config.yaml --smoke
uv run pytest -q
```

On a ROCm machine:

```
uv sync --extra rocm --extra data
uv run nightlab data fineweb-edu                      # once, ~300M tokens
uv run nightlab train experiments/baseline/config.yaml
```

A result on a different GPU gets its own baseline and is never compared against the reference card. [Hardware notes](docs/hardware.md). [Runner setup](docs/runner-setup.md).

## Layout

```
nightlab/           library: config, model, components/, train, data, ledger, render
experiments/        NNNN-slug/config.yaml + README.md, chronological queue
ledger/             results.jsonl + schema
radar/              sources.yaml, rubric.md
reports/            nights/ (agent logs) and weekly/ (human-edited digests)
.claude/routines/   the six agent prompts
.github/workflows/  ci.yml on CPU, gpu-experiment.yml on the self-hosted runner
docs/               budget, hardware, runner setup
```

## Status

Night 1. The baseline has run once on the reference hardware and is in the ledger. Next: a second baseline seed to measure noise, then the first queued experiment, Muon.

MIT. Built by [Alex Rossie](https://github.com/alex-rossie) with Claude Code doing the night shift.
