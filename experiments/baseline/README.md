# baseline

The reference model. About 25M non-embedding parameters, which is small enough that a 600 s run on the reference GPU sees a few hundred million tokens and lands in the regime where architecture and optimizer changes are known to show up.

Nothing clever on purpose. If a component in here is not the boring default, that is a bug.
