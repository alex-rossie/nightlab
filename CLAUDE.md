# nightlab

An autonomous research lab in a repo. Agents read new LLM papers nightly, reimplement one at small scale, run it for a fixed 600 s on one consumer AMD GPU under ROCm, and publish whether it replicated. Read this file before doing anything.

## The rules that matter

1. **The ledger is the truth.** `ledger/results.jsonl` is the only source for any number in any prose. Never edit a ledger entry. Never write a result into prose that is not in the ledger. The README scoreboard is rendered, not written.
2. **Humans set verdicts.** An agent submits results as `pending` and may propose a verdict in a PR body. Only a human runs `nightlab ledger-add` with a verdict.
3. **Criterion before result.** Every experiment README states a numeric replication criterion before the GPU run. If you find yourself writing the criterion after seeing the number, stop.
4. **Wall clock is the budget.** 600 s including compile. Do not change `train.wall_seconds` in an experiment config. Do not add step or token limits.
5. **One change per experiment.** A new registered component or a config diff against `experiments/baseline`. Never modify baseline components in place; add a new one.
6. **Clean commits only.** The ledger schema rejects dirty or unknown commit hashes.
7. **Failures are content.** A night where the implementer got stuck, or a paper that did not replicate, gets written up with the same care as a success. Say what you guessed and what you were unsure about.

## Layout

```
nightlab/            library: config, model, components/, train, data, ledger, render, cli
experiments/         one dir per experiment, NNNN-slug, config.yaml + README.md
ledger/              results.jsonl and its JSON schema
radar/               sources and rubric for the nightly paper scan
reports/nights/      one file per routine per night, including failures
reports/weekly/      human-edited weekly digests
.claude/routines/    the prompts for each scheduled agent
.github/workflows/   ci.yml (CPU) and gpu-experiment.yml (self-hosted ROCm runner)
docs/                budget, hardware, runner setup
```

## Commands

```
uv sync --extra cpu --extra dev            # laptop / CI
uv sync --extra rocm --extra data          # lab machine
uv run nightlab train experiments/baseline/config.yaml
uv run nightlab train experiments/baseline/config.yaml --smoke
uv run nightlab data fineweb-edu
uv run nightlab ledger-add runs/<id>/result.json --verdict replicated
uv run nightlab ledger-validate
uv run nightlab render --check
uv run ruff check . && uv run pytest -q
```

## Style

Python 3.12, ruff with a 100-column limit, type hints on public functions, no comments that restate the code. Prose in READMEs and reports: short sentences, numbers in tables, no hype words. Cite run_ids when referring to results.

## Labels

`queued` candidate issue · `experiment` PR with a new experiment · `needs-gpu` run it on the reference hardware · `gpu-done` result posted · `stuck` implementer gave up, needs a human · `chore` mechanical, may auto-merge · `digest` weekly writeup · `keep` do not auto-close.
