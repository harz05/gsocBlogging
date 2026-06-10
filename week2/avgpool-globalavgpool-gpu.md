# AvgPool and GlobalAveragePool GPU

- PR: [ML4EP/SOFIE #27](https://github.com/ML4EP/SOFIE/pull/27) (OPEN), extended from MaxPool to
  all pool modes
- File: `core/inc/SOFIE/ROperator_Pool.hxx`
- Builds on [../week1/maxpool-gpu.md](../week1/maxpool-gpu.md)

The MaxPool GPU work (#27) added the first GPU code generation for `ROperator_Pool`, but only for
MaxPool: each of the three `Generate_GPU_*` methods returned early for any other mode. AveragePool
and GlobalAveragePool still emitted nothing on the GPU path. This extends the same PR to cover
them, so the whole Pool operator runs on GPU.

## What the change does

The three GPU methods now handle `AveragePool` as well as `MaxPool`. The index bookkeeping
(decoding the flat output index into coordinates, the base offset, the in-bounds checks) is
identical between the two, so only three small fragments differ, and they are produced by local
lambdas in the kernel generator:

- init: `value = -INFINITY` for max, `value = 0` for average (plus a run-time `count` when needed).
- accumulate: keep-the-larger for max, `value += X[...]` for average (plus `++count`).
- finalize: nothing for max, `value /= divisor` for average.

### The divisor, matched to the CPU operator

The average divisor follows the CPU `Generate()` in the same file exactly:

- `count_include_pad == 0` (the ONNX default) and padding present: padded cells are excluded, so
  the divisor is the number of in-bounds cells, counted at run time.
- otherwise (`count_include_pad == 1`, or no padding): the divisor is the constant kernel area
  (`kh`, `kh*kw`, or `kh*kw*kd`).

The only differences from the CPU reference are required or cosmetic: `float(nsum)` becomes
`static_cast<T>(count)` because the GPU kernel is templated on `T` while the CPU operator is float
only, and the counter is named `count` instead of `nsum`.

### GlobalAveragePool comes for free

`Initialize` already rewrites GlobalAveragePool into an AveragePool whose kernel equals the image
size and whose pads are all zero. So it reaches the GPU methods as a plain AveragePool, takes the
constant-divisor path, and divides by the full image area. No separate kernel is needed.

### No MaxPool regression

When the mode is MaxPool the emitted text (struct name, member name, comments, body) is identical
to before the change, so the existing MaxPool1d/2d/3d tests stay valid.

## Tests

Four GTests in `core/test/TestCustomModelsFromONNXForAlpakaCuda.cxx`, same pattern as the MaxPool
tests (run the generated GPU code, compare to a reference within tolerance, iota inputs so a wrong
divisor or region is visible):

| Test | Model | Path it covers |
|---|---|---|
| `AvgPool` | reuses the existing `AvgPool.onnx` | no padding, constant divisor, checked against the trusted CPU reference |
| `AvgPoolPad` | new `AvgPoolPad.onnx` | kernel 3x3, pad 1, count_include_pad 0, the run-time count path |
| `AvgPoolCountIncludePad` | new `AvgPoolCountIncludePad.onnx` | same model, count_include_pad 1, constant kernel-area divisor |
| `GlobalAvgPool2d` | new `GlobalAvgPool2d.onnx` | GlobalAveragePool, channel mean |

The first case reusing the model already in the repo is the stronger check, the CPU reference for
it is already trusted. Padding in the new models is symmetric, so they isolate the divisor logic
from the begin/end pad layout (the asymmetric-pad bug handled in [pool-padfix-port.md](pool-padfix-port.md)).

## Result

All 89 alpaka GTests pass on Colab T4, including the four new ones. The `AvgPoolPad` pass confirms
the run-time count divisor is correct on GPU, the part most likely to hide a bug.
