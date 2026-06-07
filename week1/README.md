# Week 1 (2026-05-25 to 05-31)

The first coding week. The theme was Conv GPU correctness and the first GPU operator beyond what
already existed. Most of the week's value came from tests: writing the first tests for the
grouped Conv path immediately caught a regression that had shipped into `gpu/alpaka`, and a
parallel investigation found the same untested-path class of bug in the CPU Pool operator
upstream.

## What happened

1. Implemented the groups>1 tests proposed in community bonding (PR #22). On the restructured
   `gpu/alpaka` HEAD they failed, which exposed a real regression.
2. Diagnosed and fixed that regression: grouped Conv was computing all output channels for every
   group because `gemm_n` was never divided by the group count (issue #23, PR #24).
3. Proposed moving the Conv weight-vectorisation kernel out of `infer()` into session init,
   since it produces an invariant result (issue #25, the A1 optimization). Implementation lands
   in Week 2.
4. Added the first GPU implementation of MaxPool (1D/2D/3D) to the fork (issue #26, PR #27).
5. Found and fixed an asymmetric-padding bug in the upstream CPU Pool operator (root issue
   #22436, PR #22438, merged the same day). This is the third instance of the untested-path bug
   class.

## Docs in this week

- [conv-group-tests.md](conv-group-tests.md) — PR #22, the groups>1 tests
- [grouped-conv-regression.md](grouped-conv-regression.md) — issue #23 + PR #24, the regression
  the tests caught and its fix
- [weightvec-init-optim.md](weightvec-init-optim.md) — issue #25, the A1 optimization proposal
- [maxpool-gpu.md](maxpool-gpu.md) — issue #26 + PR #27, MaxPool GPU
- [root-pool-asym-pad-bug.md](root-pool-asym-pad-bug.md) — root issue #22436 + PR #22438

See [../STATUS.md](../STATUS.md) for live status and [../BUILD-NOTES.md](../BUILD-NOTES.md) for
build and test workflows.
