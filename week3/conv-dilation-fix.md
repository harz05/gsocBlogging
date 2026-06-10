# Conv dilation>1 fix, GPU and CPU (issue #32, PR #34)

Fork issue: https://github.com/ML4EP/SOFIE/issues/32
Fork PR: https://github.com/ML4EP/SOFIE/pull/34 (branch `fix/conv-dilation`)

This is the fork-side follow-up to the ROOT dilation bug (root #22473 / #22474, written up in
[../week2/conv-dilation-bug.md](../week2/conv-dilation-bug.md)). The same double-count exists on
the fork, in both the CPU path and the GPU path, and neither was fixed.

## Root cause (same as ROOT)

In `Initialize`, `fAttrKernelShape` is overwritten with the dilation-expanded size
`k + (dilation-1)*(k-1)` (a 3x3 kernel with dilation 2 becomes 5x5). That expanded value is then
used again together with `fAttrDilations`, so dilation is counted twice. With dilation 1 the
expansion is a no-op, which is why no existing test caught it.

## Two different symptoms

- **CPU path**: emits `Im2col(..., fAttrKernelShape[0], fAttrKernelShape[1], ..., fAttrDilations[0], fAttrDilations[1], ...)`
  - the expanded kernel (5,5) together with dilation (2,2). Im2col applies dilation itself, computes a
  negative output dimension, writes past the buffer and segfaults. Exactly root #22473.
- **GPU path**: the weight-vec kernel folds the gap into the dilated `_f` layout (`kh*10u + kw*2u`),
  and the im2col kernel re-applies it: it decodes rows with the raw width (`k_rem / 3u`) while the
  matrix is sized with the expanded kernel, and gathers at `kw*2u`. The two no longer line up, so
  the result is silently wrong - no crash, because the output dimensions stay consistent.

## The fix

The non-obvious part: on CPU `_f` is a `std::vector` (zero-initialised for free), but on GPU it is
an alpaka buffer that is not zeroed, and the dilated `_f` layout has holes. So the GPU needs more
than the CPU one-liner.

- **CPU** (one line, ports root #22474): reset `fAttrDilations = {1,1,1}` after the weight reorder,
  so the dense im2col below does not re-apply dilation.
- **GPU** (three coordinated changes, all in `Generate_GPU_Kernel_ALPAKA` / `Generate_GPU_ALPAKA`,
  no `Initialize` change):
  1. im2col decodes rows over the effective kernel extents (`fAttrKernelShape`) instead of the raw
     width, so columns line up with the dilated `_f` slots.
  2. im2col gather drops the re-applied dilation (`oh*stride + kh`, not `kh*dilation`).
  3. `_f` is zeroed (`alpaka::memset`) before the weight-vec kernel when dilation>1, so the holes
     stay 0. Guarded so it is a no-op at dilation 1.

This mirrors ROOT's "expanded kernel + dilation 1" approach on the GPU.

## Verification

Test `ConvWithDilation`: x[1,1,7,7], W[2,1,3,3] iota, 3x3 kernel, dilation 2, no padding, reference
from PyTorch `F.conv2d(dilation=2)`. On Colab T4:

- with the fix: `ConvWithDilation` passes, all existing Conv tests still pass (no regression - the
  changes are no-ops at dilation 1).
- reverting only the codegen back to `gpu/alpaka` and rebuilding: `ConvWithDilation` FAILS. That
  red->green is the before/after proof for #32.

The generated im2col, before -> after: decode `k_rem / 3u -> k_rem / 5u`, gather
`oh*1u + kh*2u -> oh*1u + kh`, plus the added `alpaka::memset`.
