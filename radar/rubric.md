# Radar rubric

A candidate paper or release gets queued as an experiment only if it clears all four gates. Score each 0 to 2 and require a total of at least 6 with no zeros.

1. **Testable at the budget.** The claim is about pretraining loss, throughput, or a component swap that a 25M-parameter model trained for 600 s can meaningfully exercise. Claims that only appear above 1B parameters, or that need RL, long-context eval, or instruction data, score 0 here and get a note in the weekly digest instead.
2. **Isolatable.** The change can be expressed as one registered component or one config diff against `baseline`. If it needs three simultaneous changes, it is three experiments or none.
3. **Falsifiable.** The paper states an effect size, or one can be inferred well enough to write a replication criterion before running.
4. **Worth knowing.** Someone building or training models this year would change a decision based on the answer. Novelty alone is not enough; a boring result about a widely adopted technique beats a flashy result about a niche one.

Ties go to the candidate with the shortest implementation. Every queued experiment gets an issue using the paper template with the four scores shown.
