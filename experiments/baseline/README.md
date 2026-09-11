# baseline

The reference model. About 25M non-embedding parameters, 77M with the untied embedding and output head; the head alone costs about as many FLOPs per token as the eight blocks, so the model is about twice as expensive per token as its 25M body suggests. A 600 s run on the reference GPU sees about 80M tokens, still the regime where architecture and optimizer changes are known to show up.

**Status.** Baseline recorded. Run `baseline-20260910T214203Z-5f6e26`: val loss 4.303 at 600 s, 2474 steps, 81M tokens, 135k tokens/s, 15.6 GB peak. Single seed; a second seed is queued to measure noise. The training loop's final evaluation changed after this run (whole validation slice in fixed windows, instead of 40 sampled batches), so `nightlab compare` and `ledger-add` refuse to score any new run against it. The reference has to be re-run, two seeds, before the next verdict.

Nothing clever on purpose. If a component in here is not the boring default, that is a bug.
