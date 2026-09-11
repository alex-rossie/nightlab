# Experiments

One directory per experiment. Each has a `config.yaml` and a `README.md` that states the claim, the source, what was changed, and what result would count as a replication. Write the success criterion **before** the run.

Naming: `NNNN-short-slug`. The number is assigned when the experiment is queued, so the directory listing is the chronological queue.

The `baseline/` experiment is the reference every other run is compared against. It is re-run whenever the training loop, data, or dependencies change in a way that could shift the number. Each verdict in the ledger records the baseline run it was judged against.

## Budget

| Setting | Value |
|---|---|
| Wall clock | 600 s including compile |
| Hardware | one consumer AMD GPU, 24 GB, ROCm |
| Data | FineWeb-Edu sample-10BT, first 300M tokens, fixed 10M-token val slice |
| Metric | validation loss (nats per GPT-2 token) over the whole val slice, in fixed non-overlapping 1024-token windows; curve points use its first 20 batches |
| Compile | cold on every run, inside the budget |
| Seeds | one seed by default; a `replicated` verdict requires a second seed |

See `docs/budget.md` for why these choices.
