# Conv GPU groups > 1 tests

- PR: [ML4EP/SOFIE #22](https://github.com/ML4EP/SOFIE/pull/22) (OPEN), closes issue
  [#21](https://github.com/ML4EP/SOFIE/issues/21)
- Files: `core/test/TestCustomModelsFromONNXForAlpakaCuda.cxx` plus three ONNX models and their
  reference outputs

Implements the groups>1 test plan raised in community bonding (see
[../community-bonding/conv-gpu-untested-paths.md](../community-bonding/conv-gpu-untested-paths.md)).
The grouped branch of `Generate_GPU_ALPAKA` was emitted but never code-generated, because no
test model had `group > 1`.

## The three tests

- **ConvGroup2** — input `[1, 4, 5, 5]`, weight `[4, 2, 3, 3]`, groups=2.
- **ConvGroup4** — input `[1, 4, 5, 5]`, weight `[4, 1, 3, 3]`, groups=4 (depthwise edge case).
- **ConvBatch4Group2** — input `[4, 4, 5, 5]`, groups=2. Validates that the outer batch loop and
  the inner group loop nest correctly.

Reference outputs verified against PyTorch `F.conv2d` on Colab T4.

## Design decisions

- **Weights are iota-filled, not all-ones.** With all-ones weights a bug that reads the wrong
  per-group weight slice produces identical output to the correct case, since every group's
  filter slice has the same values. Distinct values per slice make such a bug visible. Inputs
  use `std::iota` for the same reason. This is a deliberate departure from the all-ones weights
  used in the batch tests (PR #20).
- **`inC = outC = 4`** keeps the architecture simple. An earlier idea to make `inC != outC` to
  catch a hypothetical channel-stride bug was dropped because that bug class was speculative, not
  anchored to anything in the code.
- **Three tests, not more.** The minimum set to cover the four concrete bug classes: the
  per-group weight offset, the per-group input offset, the per-group output offset, and loop
  nesting. The offset formulas are linear in group and batch index, so more permutations give
  diminishing returns.

## What the tests caught

On the older `gpu/alpaka` HEAD the three tests passed. After syncing to the restructured HEAD
they failed, with output channels 0 and 1 correct and channels 2 and 3 wrong. That is a real
regression, written up in [grouped-conv-regression.md](grouped-conv-regression.md). The tests
did exactly their job: exercising a path that had never run and catching a bug the absence of
tests had hidden.

Note on the restructure: between the first and second test runs, upstream renamed
`src/SOFIE_core/...` to `core/...`, so the original PR branch pointed at files that no longer
existed. The branch was recreated fresh from the new HEAD with the ONNX and reference files
regenerated at the new paths. See [../BUILD-NOTES.md](../BUILD-NOTES.md) for the exFAT
`git config core.fileMode false` fix that removed the mode-flip noise during this.
