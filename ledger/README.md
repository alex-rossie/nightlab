# The ledger

`results.jsonl` is the single source of truth. One line per completed run. The README scoreboard is rendered from it by `uv run nightlab render` and CI fails if the two disagree.

Rules:

- A run enters the ledger only from a clean commit. The schema rejects `-dirty` and `unknown`.
- The agent that ran an experiment submits it as `pending`. A human sets the verdict on the PR. The agent may propose a verdict in the PR body, never in the ledger.
- Verdicts: `replicated` (effect direction and rough size match the claim), `partial` (direction matches, size does not, or it only holds in some settings), `failed` (no effect or wrong direction at this scale), `inconclusive` (run was noisy, broken, or the claim is not testable at this budget), `baseline` (reference run).
- Entries are never deleted. A superseded result is left in place and the newer run sits above it.
- The comparison baseline for any run is the most recent `baseline` entry on the same hardware string.
