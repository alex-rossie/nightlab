# 2026-09-10 inaugural run

Launched by hand, not by a routine. The routines are not scheduled yet.

What happened: ROCm torch 2.14 installed from the PyTorch rocm7.2 index with no system ROCm. The GPU was detected on the first try. A 300M-token FineWeb-Edu slice was tokenized in a few minutes. The baseline ran for 600 s from a clean commit and produced run `baseline-20260910T214203Z-5f6e26`.

| | |
|---|---|
| val loss | 4.303 |
| steps | 2474 |
| tokens | 81M |
| tokens/s | 135k |
| peak memory | 15.6 GB |
| compile | included, roughly the first 15 s |

Loss curve at one-minute intervals: 6.00, 5.37, 5.04, 4.83, 4.67, 4.55, 4.46, 4.40, 4.35, then 4.30 at the end. Still falling steadily at 600 s, so the budget is short of convergence by design.

Things noticed: the ROCm runtime prints two `(null): No such file or directory` lines to stderr on import. Harmless so far, not investigated. Peak memory leaves headroom for a larger batch, which is a knob deliberately left alone since the baseline should stay boring.

Not done: a second seed. Nothing in the scoreboard should be read as significant until seed noise is measured.
