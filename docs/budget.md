# Why a fixed wall-clock budget

Every experiment in this repo gets exactly 600 seconds on one consumer AMD GPU running ROCm. Not a fixed number of steps, not a fixed number of tokens. Seconds.

**It is the honest unit.** A technique that halves loss per step but doubles step time has not helped anyone. Papers report per-step or per-token curves because that is what is convenient to plot. Practitioners pay in GPU hours. Wall clock is the number that matters and it is the number this lab reports.

**It makes throughput a first-class result.** Every ledger row has tokens per second next to validation loss. A change that costs speed has to buy it back in loss within the same ten minutes. Compile time counts, and every run compiles cold: the training loop points the compiler cache at an empty directory, so no run inherits kernels from an earlier one. From this change on, the ledger records how long the first step and the first evaluation took, and refuses to compare two runs that were not scored the same way.

**It is reproducible by anyone with the same class of card.** A used consumer RDNA card is a few hundred dollars. The runs are not cheap in the sense of being free, but they are cheap in the sense that anyone can check them.

**It keeps the agent honest.** An autonomous implementer cannot game a wall-clock budget by fiddling with batch size or step counts. It can only make the model learn faster, or not.

## What the budget cannot test

Plenty. Anything that only emerges above a billion parameters. Anything about long context beyond 1024 tokens. Anything about instruction following, RL, or evals other than next-token loss. Those claims score zero on the first radar gate and are listed in the weekly digest as "out of scope, not disproven".

## What counts as a replication

The criterion is written in the experiment README before the run and must contain a number. The default bar is a val-loss difference of at least 0.02 nats against the current baseline on the same hardware. That bar assumes single-seed noise near 0.01, which is an assumption until two baseline seeds scored under the same evaluation are in the ledger; the bar moves if the measured spread says it should. A `replicated` verdict needs a second seed, and the ledger refuses one without it. A `partial` verdict is for the right direction with a smaller size than claimed. `failed` is no effect or wrong direction. `inconclusive` is reserved for runs that were noisy or broken, and it is a valid and common outcome.
