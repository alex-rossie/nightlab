# 0001-muon

**Claim.** Muon, an optimizer that orthogonalizes the momentum update of each 2D hidden weight matrix with a Newton-Schulz iteration, trains small transformers faster than AdamW at equal compute. Source: Keller Jordan's writeup and the modded-nanogpt speedrun record history. Later scaled up in Moonshot's Kimi K2 training.

**What changed.** `optim.name: muon`. Hidden matrices (attention qkv/proj, MLP up/gate/down) get Muon at lr 0.02, momentum 0.95, Nesterov. Embeddings, the output head, and norm weights stay on AdamW with the baseline settings. Everything else identical to `baseline`.

**Replication criterion, written before the run.** Val loss at 600 s at least 0.05 nats below the baseline on the same night. The speedrun history suggests a larger gap at this scale, so anything between 0.02 and 0.05 is `partial`. Note that Newton-Schulz adds per-step cost, so the tokens/s column must be reported alongside loss; a per-step win that loses at wall clock is a `failed`.

**Status.** Queued. Not yet run on the reference hardware.
