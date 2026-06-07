# MaxPool/AvgPool wrong pad indices for asymmetric padding (upstream ROOT)

- Issue: [root-project/root #22436](https://github.com/root-project/root/issues/22436) (CLOSED)
- PR: [root-project/root #22438](https://github.com/root-project/root/pull/22438) (MERGED 2026-05-31)
- File: `tmva/sofie/inc/TMVA/ROperator_Pool.hxx`

A CPU-side SOFIE bug in upstream ROOT, found while reading the Pool operator. Same untested-path
class as the others. Found, fixed, and merged the same day.

## The bug

`ROperator_Pool::Generate()` computes the pooling-window start and stop bounds (`hmin/hmax`,
`wmin/wmax`, `dmin/dmax`). It read the end pads at fixed indices `1, 3, 5`, assuming pads were
laid out interleaved as `[begin, end, begin, end, ...]`. But ONNX, and SOFIE's parser verbatim,
store pads in grouped order `[begin0, begin1, ..., end0, end1, ...]`. So the begins live at
`0, 1, 2` and the ends at `fDim, fDim+1, fDim+2`, where `fDim` is the number of spatial axes.

For padding that is not symmetric across the spatial axes, the code swapped, for example, the
height-end pad with the width-begin pad. That produces wrong window positions and even a wrong
output shape, silently, with no error or throw.

It was masked because every existing pool test in both ROOT and the fork uses zero padding,
where the wrong index still reads 0. Conv reads the same attribute correctly (its
`ConvWithAsymmetricPadding` test passes), which confirmed grouped is the intended layout.

## The fix

In `Generate()`, read begins at `0, 1, 2` and ends at `fDim, fDim+1, fDim+2`:

| bound | before | after |
|---|---|---|
| hmin | `-fAttrPads[0]` | unchanged (begin, correct) |
| hmax | `fAttrPads[1]` | `fAttrPads[fDim]` |
| wmin | `-fAttrPads[2]` | `-fAttrPads[1]` |
| wmax | `fAttrPads[3]` | `fAttrPads[fDim+1]` |
| dmin | `-fAttrPads[4]` | `-fAttrPads[2]` |
| dmax | `fAttrPads[5]` | `fAttrPads[fDim+2]` |

With all-zero pads nothing changes, so existing tests stay green. The 1D case was never wrong,
because with one axis the begin and end are already adjacent.

A subtlety: `fAttrPads.resize(6, 0)` only appends zeros, it does not re-slot into a 3D layout. A
2D pad keeps its grouped form `[h_begin, w_begin, h_end, w_end]` plus trailing zeros, so `h_end`
sits at index `fDim` (= 2 for 2D), exactly where shape inference reads it.

## Verification

The bug is in code generation, so it is visible in the generated `.hxx` without running
inference. Generated SOFIE code from an asymmetric model (`MaxPool 2x2 stride1 pads=[0,1,0,1]`)
and read the emitted bound constants: buggy emits `hmax=4, wmin=0`; correct is `hmax=3,
wmin=-1`. Cross-checked the generated loop bounds against PyTorch (with manual `-inf` padding so
asymmetric pads work) across 2D/3D and various strides: the new bounds pass every case, the old
bounds fail every genuinely-asymmetric case. The PR adds the regression test
`MaxPool2d_AsymPad` (input `1x1x4x4` iota, `pads=[0,1,0,1]`, PyTorch-computed reference, output
shape 3x5).

Proved the test guards the bug by stashing the fix, rebuilding the codegen tool, and running
ctest: exactly one test fails (`ONNX.MaxPool2d_AsymPad`), the rest pass.

## Port to the fork

The same bug exists in the fork's Pool operator. The CPU port is PR #28 (see
[../week2/pool-padfix-port.md](../week2/pool-padfix-port.md)). The new GPU MaxPool kernel
(PR #27, [maxpool-gpu.md](maxpool-gpu.md)) needs the same correction as a follow-up.

Build and verification details, including the exFAT and stale-`.pcm` traps hit while building
ROOT for this, are in [../BUILD-NOTES.md](../BUILD-NOTES.md).
