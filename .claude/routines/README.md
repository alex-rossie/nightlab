# Routines

These prompts are the lab's staff. Each one is a scheduled Claude Code agent. The prompts are checked in so that the maintenance system is reviewable, diffable, and part of the showcase.

| Routine | Schedule (Pacific / UTC) | Model | Runs where | Can merge? |
|---|---|---|---|---|
| `radar.md` | nightly 02:00 / 09:00 | Sonnet 5 | cloud | no, opens issues only |
| `implement.md` | nightly 03:00 / 10:00 | Opus 5 | cloud, then GPU runner via label | no, opens a draft PR |
| `housekeeping.md` | nightly 05:00 / 12:00 | Sonnet 5 | cloud | mechanical PRs only, on green CI |
| `ledger-sync.md` | nightly 06:00 / 13:00 | Sonnet 5 | cloud | yes, render-only commits |
| `weekly-digest.md` | Sunday 08:00 / 15:00 | Sonnet 5 | cloud | no, opens a PR |
| `weekly-audit.md` | Sunday 09:00 / 16:00 | Opus 5 | cloud | no, opens issues |

Cron is fixed in UTC, so the Pacific times move one hour earlier when daylight saving ends.

The human's counterpart is the `/lab` skill in `.claude/skills/lab/`, which handles review, local GPU runs, verdicts, and baseline refreshes.

Cloud routines have no GPU. Anything that needs the reference hardware goes through the `needs-gpu` PR label, which a human applies in the GitHub UI and the self-hosted runner picks up once that human also approves the `gpu` deployment. Routines never do either; a hook in `.claude/hooks/` blocks the obvious `gh` forms of both from Bash in any session opened in this repo. See `docs/runner-setup.md`.

Each routine is registered as a Claude Code cloud routine whose message is a one-line bootstrap: read CLAUDE.md, then follow the named prompt file in this directory. The prompt files are the source of truth; editing one changes the routine's behavior on its next run without re-registering. Register or inspect them from a Claude Code session with `/schedule`. Never run `housekeeping.md` on the lab machine: its `uv sync --extra cpu --extra dev` would replace the ROCm torch in the checkout's venv. To run one of the others locally instead:

```
claude -p "$(cat .claude/routines/radar.md)"
```

Budget caps live in the prompts and in the runner workflow. If a routine hits a cap it stops and writes a note to `reports/nights/`, it does not retry.
