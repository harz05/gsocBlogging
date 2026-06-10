# Conv GPU SAME_UPPER autopad test (issue #33, PR #35)

Fork issue: https://github.com/ML4EP/SOFIE/issues/33
Fork PR: https://github.com/ML4EP/SOFIE/pull/35 (branch `feat/conv-autopad-upper-test`)

The Conv alpaka tests covered the `SAME_LOWER` autopad branch but not `SAME_UPPER`. The codegen
handles both in `Initialize`, so `SAME_UPPER` was generated but never exercised.

## The analysis that shaped the test

`SAME_UPPER` and `SAME_LOWER` differ only in how leftover padding splits: one puts the extra pad at
the end, the other at the start. When the total padding is **even** they place identical pads and
give identical output; they only differ when the total padding is **odd**.

ROOT's portable `ConvWithAutopadSameUpper` model (input 5, kernel 3, stride 1) has total padding 2
(even), so for that model `SAME_UPPER` equals `SAME_LOWER`. Porting it as-is would only cover the
code branch, not the upper-specific behaviour - it would pass even if the split were wrong.

So the test uses an **odd**-padding model: x[1,1,4,4], kernel 3, stride 2 -> total pad 1 ->
`SAME_UPPER` pads begin 0 / end 1, `SAME_LOWER` pads begin 1 / end 0. Output `[45, 39, 66, 50]`
(verified vs PyTorch with explicit end-heavy padding); `SAME_LOWER` would give `[10, 24, 51, 90]`,
so the test genuinely distinguishes the two.

The generated im2col confirms it at the codegen level: `SAME_UPPER` emits `... - 0` (no pad at the
start), `SAME_LOWER` emits `... - 1`.

## Why this case matters in practice

Odd-split is not a corner case. A 3x3 stride-2 conv (the canonical downsampling layer) on any
even-sized feature map (28, 32, 56, 112, 224 ...) gives odd total padding. And TensorFlow/Keras
"SAME" padding maps to ONNX `auto_pad=SAME_UPPER` (extra pad at the bottom/right), so TF/Keras
strided convs routinely rely on the `SAME_UPPER` placement. This was raised as a comment on the
issue before settling on the odd-pad model.

GPU only: the fork builds the alpaka test suite, so no CPU test (the CPU SOFIE tests live in ROOT).
