# Conv GPU: the emitted-but-untested batch and group paths

- PR: [ML4EP/SOFIE #20](https://github.com/ML4EP/SOFIE/pull/20) (OPEN) — tests for batch > 1
- Issue: [ML4EP/SOFIE #21](https://github.com/ML4EP/SOFIE/issues/21) (OPEN) — tests for groups > 1
- File: `core/inc/SOFIE/ROperator_Conv.hxx`, tests in
  `core/test/TestCustomModelsFromONNXForAlpakaCuda.cxx`

The Week 1-2 proposal deliverable is "Conv batch>1 and group convolution, GTests passing." By
the time coding started, the underlying GPU code already existed upstream on `gpu/alpaka`
(Sanjiban had implemented the batch and group logic in commit `dae834e`). So the first job was
not to write the operator, it was to verify it, because neither path had ever been run.

## What was untested

`Generate_GPU_ALPAKA` in `ROperator_Conv.hxx` emits a batch loop that offsets the input and
output pointers per sample and runs im2col + GEMM for each one. But every Conv ONNX model in
`test/input_models/` has input shape `[1, 1, H, W]`, so the loop body always ran exactly once.
The multi-sample path was never exercised.

The same was true for grouped convolution: `Generate_GPU_ALPAKA` has a per-group im2col + GEMM
loop that partitions input channels and weights into G groups, but every Conv ONNX model has
`group = 1` (verified by decoding each ONNX with the Python `onnx` library and reading the Conv
node attributes), so the grouped branch was never code-generated.

## PR #20 — batch > 1 tests

Three tests, `ConvBatch2`, `ConvBatch4`, `ConvBatch8`, following the existing TEST_F pattern.
Inputs use `std::iota`, weights are all-ones, matching the architecture of the existing single
Conv test so the test exercises the batch loop rather than the Conv math itself. Reference
outputs verified against PyTorch `F.conv2d`. All pass on Colab T4, 60 tests green, no
regressions. PR #20 closes the earlier batch-test issue #18.

This PR was opened on top of the older `gpu/alpaka` HEAD and sat open while Sanjiban worked on a
project restructure, so it needs a rebase once that settles.

## Issue #21 — groups > 1 tests

Raised following the same pattern: the grouped path is emitted but no model has `group > 1`, so
it has never been tested. The plan in the issue is to add models with groups 2 and 4 plus a
combined batch>1 + groups>1 model (for example batch=4, groups=2) to validate that the outer
batch loop and inner group loop nest correctly. The implementation of those tests lands in
Week 1 as PR #22 (see [../week1/conv-group-tests.md](../week1/conv-group-tests.md)), and it
immediately caught a real regression (see
[../week1/grouped-conv-regression.md](../week1/grouped-conv-regression.md)).

## Test design choices worth remembering

- For the group tests, weights are `iota`-filled rather than all-ones. With all-ones weights a
  bug that reads the wrong per-group weight slice produces identical output to the correct case,
  because every slice has the same values. Distinct values per slice make such a bug visible.
- Keeping `inC = outC` and a small channel count keeps the architecture simple while still
  covering the offset and loop-nesting bug classes.

See [../BUILD-NOTES.md](../BUILD-NOTES.md) for the Colab build and how to run a single gtest
case with `--gtest_filter`.
