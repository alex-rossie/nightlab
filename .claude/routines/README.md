# Routines

These prompts are the lab's staff. Each one is a scheduled Claude Code agent. The prompts are checked in so that the maintenance system is reviewable, diffable, and part of the showcase.

| Routine | Schedule (UTC) | Runs where | Can merge? |
|---|---|---|---|
| `radar.md` | nightly 02:00 | cloud | no, opens issues only |
| `implement.md` | nightly 03:00 | cloud, then GPU runner via label | no, opens a draft PR |
| `housekeeping.md` | nightly 05:00 | cloud | mechanical PRs only, on green CI |
| `ledger-sync.md` | nightly 06:00 | cloud | yes, render-only commits |
| `weekly-digest.md` | Sunday 08:00 | cloud | no, opens a PR |
| `weekly-audit.md` | Sunday 09:00 | cloud | no, opens issues |

Cloud routines have no GPU. Anything that needs the reference hardware goes through the `needs-gpu` PR label, which the self-hosted runner picks up. See `docs/runner-setup.md`.

Register a routine from a Claude Code session with `/schedule`, pasting the prompt file's contents. To run one locally instead:

```
claude -p "$(cat .claude/routines/radar.md)"
```

Budget caps live in the prompts and in the runner workflow. If a routine hits a cap it stops and writes a note to `reports/nights/`, it does not retry.
