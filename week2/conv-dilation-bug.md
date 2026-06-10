# Conv dilation > 1 crashes (double-counted dilation in im2col)

- Issue: [root-project/root #22473](https://github.com/root-project/root/issues/22473) (OPEN)
- PR: [root-project/root #22474](https://github.com/root-project/root/pull/22474) (OPEN), +29 -0
- File: `tmva/sofie/inc/TMVA/ROperator_Conv.hxx`

A CPU-side SOFIE bug in upstream ROOT. Same untested-path class as the others: dilation > 1 had
no test anywhere in ROOT or the fork, and every existing Conv test uses dilation 1 where the bug
is harmless. With dilation > 1 the generated code segfaults.

## How the dilated path is meant to work

ROOT intentionally builds a dilated `_f` weight buffer. The weight-reorder loop walks the
original kernel (for a 3x3 kernel, the 9 real taps) and scatters those 9 weights into a 5x5
dilated layout using dilation strides. So `gemm_k` is set to `inC * product(expanded kernel)`,
which is 25 for a 3x3 kernel at dilation 2, to match the dilated `_f`. The expanded kernel is by
design.

## The bug

`ROperator_Conv.hxx` overwrites `fAttrKernelShape` with the dilated effective size
`k + (d-1)(k-1)` (3 becomes 5 for d=2). That inflated value is then passed to `UTILITY::Im2col`
as the kernel size, and `fAttrDilations` is passed to im2col as the dilation as well. But im2col
already applies dilation itself (its receptive field is `d*(kernel-1)+1`). So dilation is
counted twice: once in the inflated kernel shape, once in the dilation argument. im2col's
internal output size then disagrees with the allocated output buffer, giving an out-of-bounds
read and write, and the generated code segfaults.

It only triggers for dilation > 1. For dilation 1 the inflated kernel equals the original, so it
is harmless, which is why every existing test passed.

## The fix

With the dilated `_f` buffer and the expanded kernel shape already in place, im2col must gather a
dense patch, so its dilation argument should be 1, not `fAttrDilations`. One line, after the
weight-reorder loop:

```cpp
fAttrDilations = std::vector<size_t>(3, 1);
```

This makes all six im2col emission sites (1D, 2D, 3D, each grouped and non-grouped) emit
dilation 1. The dilation stride locals used by the weight reorder are computed before this line,
so they are unaffected. Verified alignment: a dense im2col row at `(2*kh, 2*kw)` gathers input at
`(oh + 2*kh, ow + 2*kw)`, which is exactly where `_f` places weight `(kh, kw)`.

Note: the original issue text suggested "the kernel should be 3,3". That is wrong; reverting the
kernel inflation would break the GEMM and `_f` alignment. The issue was corrected to drop that
line. The real fix is the dilation argument, not the kernel shape.

## Verification

Added `ConvDilationModelGenerator.py` (a genuine 3x3 dilation-2 conv, iota input, numpy
dilation-aware reference), `ConvWithDilation.onnx`, `ConvWithDilation.ref.hxx`, and
`TEST(ONNX, ConvWithDilation)` in `TestCustomModelsFromONNX.cxx`. Before the fix: the five
existing Conv tests pass and `ConvWithDilation` segfaults; ctest went 67 percent (segfault).
After the fix: 100 percent pass, no regressions. Reference output (first values):
`98.4, 102.9, 107.4, 129.9, 134.4, 138.9, 161.4, 165.9, 170.4`.

Run via ctest, not the binary directly: a direct run skips the emit fixture and include-path
setup and reports file-not-found for the generated headers. See
[../BUILD-NOTES.md](../BUILD-NOTES.md).

## Port to the fork

The fork's GPU weight-vectorisation uses the same kernel inflation, so it likely has the same
double-count. Check and port after the ROOT fix lands.
