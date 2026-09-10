---
name: lab
description: The human's side of nightlab. Review experiment PRs, run one on the local GPU, set a verdict and record it in the ledger, or refresh the baseline. Use when the user is at the lab machine and says things like "review the PRs", "run PR 12", "verdict on 12", "second baseline seed".
---

You are running the `lab` skill inside the nightlab repo. Read CLAUDE.md first if you have not this session. The rules there bind you too, with one difference from the routines: you may record verdicts, but only after the user confirms one.

Arguments: `$ARGUMENTS`. The first word is the subcommand. With no arguments, run `review`.

## review

Show the user what needs a decision this morning.

1. `gh pr list --label experiment --state open --json number,title,labels,headRefName,url` and `gh issue list --label stuck --state open`.
2. For each open experiment PR, print a compact card: number, title, labels, the experiment directory it adds, the **Claim** and **Replication criterion** lines from that experiment's README, the CPU smoke line from the PR body, and whether CI is green (`gh pr checks`).
3. Read the diff (`gh pr diff N`) and say in two sentences whether the implementation matches the source described in the README. Flag anything that changes a baseline component in place, changes `train.wall_seconds`, or lacks a numeric criterion. These are hard rejects.
4. For each PR, offer exactly these choices and wait: apply `needs-gpu` (runner will pick it up), run it here now (`run N`), mark `stuck` with a comment, or skip.
5. Also list the most recent night reports in `reports/nights/` since the user's last commit, one line each, so they see where the agents struggled.

Never merge anything from this subcommand.

## run N

Run PR N's experiment at full budget on this machine. This is the fallback for when the self-hosted runner is not installed or is offline.

1. Confirm the tree is clean (`git status --porcelain` is empty). If not, stop; a dirty tree produces a ledger-ineligible result.
2. `gh pr checkout N`. Find the single changed `experiments/*/config.yaml`; if there is not exactly one, stop and say so.
3. Confirm the ROCm env: `uv sync --extra rocm --extra data` and `uv run python -c "import torch; assert torch.cuda.is_available()"`. Confirm `data/fineweb-edu/train.bin` exists, otherwise run `uv run nightlab data fineweb-edu` first and tell the user it takes a few minutes.
4. Run in the background so the session stays responsive:
   `NIGHTLAB_HARDWARE=amd-consumer-24gb uv run nightlab train <config> --out runs/pr-N --quiet`
   Do not change the budget. Do not add flags beyond `--out` and `--quiet`.
5. When it finishes, post the comment: `uv run nightlab pr-comment runs/pr-N/result.json > /tmp/c.md && gh pr comment N --body-file /tmp/c.md`, then `gh pr edit N --remove-label needs-gpu --add-label gpu-done`.
6. Print `uv run nightlab compare runs/pr-N/result.json` and go straight to `verdict N`.

## verdict N

Turn a completed run into a ledger entry. This is the step that must never be automatic.

1. Locate the result: `runs/pr-N/result.json` locally, or download the runner's artifact with `gh run download --name result-pr-N`.
2. `git checkout main && git pull`. The ledger entry's commit must be the PR's merged commit, so if the PR is not merged yet, ask the user to merge it first (`gh pr merge N --squash`), then pull. Do not merge on your own.
3. Print `uv run nightlab compare <result.json>`. Then read the experiment README's criterion and the number in it.
4. Propose one verdict with a one-sentence reason, using these definitions: `replicated` meets the criterion and, for a positive claim, a second seed on the same hardware agrees; `partial` right direction, smaller than claimed, or single-seed; `failed` no effect or wrong direction; `inconclusive` the run was noisy, broken, or hit a budget or provenance problem. If the run is single-seed and meets the bar, propose `partial` and offer to queue a second seed. Mention the tokens/s delta explicitly; a loss win bought with a large throughput loss is what the budget exists to expose.
5. Ask the user to confirm or change the verdict. Stop and wait. Do not proceed on silence.
6. On confirmation:
   `uv run nightlab ledger-add <result.json> --verdict <v> --notes "<the one-sentence reason>"`
   `uv run nightlab render`
   Update the **Status.** line in `experiments/<name>/README.md` to the verdict and run id.
   `uv run nightlab ledger-validate && uv run pytest -q`
   Commit as `ledger: <experiment> <verdict> (<run id last 6>)` and push. Close the linked issue if it is still open.

## baseline [--seed S]

Re-run the reference. Use when the audit opens a "baseline re-run needed" issue, or to add a second seed.

1. Clean tree check, ROCm env check, data check, as in `run`.
2. If `--seed` is given, run with a temporary config copy that changes only `seed`; otherwise use `experiments/baseline/config.yaml` as is. Output to `runs/baseline-<date>`.
3. Print `compare`. For a second seed, also print the spread between the two baseline val losses; this is the noise floor and belongs in the night report.
4. Ask before `ledger-add ... --verdict baseline`. Then render, update the baseline README status line, validate, commit `ledger: baseline re-run (<run id last 6>)`, push.

## Rules that apply to every subcommand

- Never edit `ledger/results.jsonl` by hand. Only `nightlab ledger-add` writes to it.
- Never change `train.wall_seconds`, and never pass `--wall-seconds` or `--smoke` for a real run.
- Never set a verdict the user has not confirmed in this session.
- Keep GPU runs in the background and check on them; do not block the session for ten minutes.
- If anything is off (dirty tree, missing baseline on this hardware, budget mismatch, criterion missing), say so and stop rather than working around it.
