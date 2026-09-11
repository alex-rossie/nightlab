# 0001-muon

**Claim.** Muon, an optimizer that orthogonalizes the momentum update of each 2D hidden weight matrix with a Newton-Schulz iteration, trains small transformers faster than AdamW at equal compute. Source: Keller Jordan's writeup and the modded-nanogpt speedrun record history. Later scaled up in Moonshot's Kimi K2 training.

**What changed.** `optim.name: muon`, nothing in the model. Hidden matrices (attention qkv and proj, MLP gate, up, and down) get Muon; embeddings, the output head, and norm weights stay on AdamW with the baseline settings (lr 6e-4, weight decay 0.1, betas 0.9/0.95). The full Muon setting, so nothing is inherited silently:

| | value | reference |
|---|---|---|
| lr | 0.02 | 0.02 |
| momentum | 0.95, Nesterov | same |
| Newton-Schulz | 5 steps, bf16, coefficients (3.4445, -4.7750, 2.0315) | same |
| update scale | sqrt(max(1, rows/cols)) per slice: 1.0 for qkv, proj, and down; sqrt(2.75) for MLP gate and up | same |
| lr warmup | 100 steps linear, from the training loop | none in the reference |
| weight decay | 0.01, decoupled; still 3.3x the baseline's per-step shrinkage of 6e-5 | 0.01 in the reference README at this lr; 0 in the reference class default; not mentioned in the blog post |
| Q, K, V | orthogonalized separately, by viewing the fused (1536, 512) weight as (3, 512, 512) | separately, per the blog post; modded-nanogpt does it with a (3, d, d) parameter |

Deviations from the source that remain, on purpose: the AdamW side keeps the baseline's lr and weight decay rather than the reference's 3e-4 and 0.01, so that the only change against `baseline` is the optimizer on hidden matrices; global gradient clipping at 1.0 is applied before the Muon step, as in the baseline; Newton-Schulz runs uncompiled, one parameter at a time (three slices at once for qkv), which costs more wall clock than the speedrun's compiled version and counts against Muon here.

An earlier draft of this experiment passed the AdamW weight decay of 0.1 into Muon, which at lr 0.02 shrinks hidden weights 33x faster per step than the baseline does. That was never run.

**Replication criterion, written before the run.** Val loss at 600 s at least 0.05 nats below the current baseline on the same hardware. The speedrun history suggests a larger gap at this scale, so anything between 0.02 and 0.05 is `partial`. Note that Newton-Schulz adds per-step cost, so the tokens/s column must be reported alongside loss; a per-step win that loses at wall clock is a `failed`.

**Status.** Queued. Not yet run on the reference hardware.
