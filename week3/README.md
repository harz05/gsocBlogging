# Week 3 (2026-06-08 to 06-14, in progress)

The third coding week. Two threads on the fork: finishing the Conv GPU test coverage (the bias,
1D, 3D, and SAME_UPPER autopad paths that were emitted but never tested) and the fork-side Conv
dilation fix. Plus two upstream ROOT items: an AveragePool ceil_mode bug and a parser gap for the
existing Swish operator, the latter found by studying the ONNX parser directly.

## What happened

1. Added GTests for three untested Conv GPU paths - bias, 1D, 3D (PR #31). Same untested-path
   class as batch and groups before them.
2. Fixed Conv dilation>1 on the fork, both the CPU path (one-line port of root #22474) and the GPU
   path (im2col decode over the effective kernel, gather without re-applied dilation, and zeroing
   `_f` since alpaka buffers are not zero-initialised). Issue #32, PR #34. Showed the before/after
   as a red->green test on Colab.
3. Added the SAME_UPPER autopad GPU test (issue #33, PR #35). The analysis here mattered:
   SAME_UPPER and SAME_LOWER are identical for even total padding, so the test deliberately uses an
   odd-padding model (input 4, kernel 3, stride 2) to actually exercise the upper split. Odd-split
   is the common TF/Keras "SAME" strided-conv case.
4. Filed an upstream AveragePool ceil_mode bug (root #22523): the overhang window that ceil_mode
   keeps is divided by the full kernel size instead of the count of real cells.
5. Studied `tmva/sofie_parsers/` and found the Swish operator is fully implemented but never
   registered in the ONNX parser, so standard ONNX Swish (opset 24) is unparseable (root #22552,
   improvement). A ReduceMax/Min lead checked out as not-implemented rather than a registration gap.

## Docs in this week

- [conv-dim-coverage-tests.md](conv-dim-coverage-tests.md) - PR #31, bias / 1D / 3D Conv GPU tests
- [conv-dilation-fix.md](conv-dilation-fix.md) - issue #32 + PR #34, the fork dilation fix (CPU + GPU)
- [conv-autopad-upper-test.md](conv-autopad-upper-test.md) - issue #33 + PR #35, SAME_UPPER and the
  even/odd padding analysis
- [avgpool-ceilmode-bug.md](avgpool-ceilmode-bug.md) - root #22523, AveragePool ceil_mode divisor
- [swish-parser-gap.md](swish-parser-gap.md) - root #22552, Swish implemented but not parser-registered

See [../STATUS.md](../STATUS.md) for live status and [../BUILD-NOTES.md](../BUILD-NOTES.md) for
build and test workflows.
