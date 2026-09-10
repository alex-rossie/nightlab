# baseline

The reference model. About 25M non-embedding parameters, which is small enough that a 600 s run on the reference GPU sees about 80M tokens and lands in the regime where architecture and optimizer changes are known to show up.

**Status.** Baseline recorded. Run `baseline-20260910T214203Z-5f6e26`: val loss 4.303 at 600 s, 2474 steps, 81M tokens, 135k tokens/s, 15.6 GB peak. Single seed; a second seed is queued to measure noise.

Nothing clever on purpose. If a component in here is not the boring default, that is a bug.
