# AveragePool ceil_mode divisor bug (ROOT #22523)

ROOT issue: https://github.com/root-project/root/issues/22523

A CPU SOFIE bug in upstream ROOT, same untested-path class as the others.

## The bug

When an ONNX AveragePool node has `ceil_mode=1`, the last sliding window can extend past the end of
the input - the "overhang" window that ceil_mode is meant to keep. SOFIE's generated code averages
such a window by dividing the sum of the real cells by the **full kernel size**, instead of by the
number of cells actually present. The overhang window comes out wrong.

In `ROperator_Pool.hxx` the divisor is computed at run time only when
`count_include_pad == 0 && doPadding`. Otherwise it is the compile-time constant kernel area. A
window can also be partial because of `ceil_mode`, which that condition does not cover:

- `count_include_pad = 0` (default) with no padding + ceil_mode -> constant divisor, wrong.
- `count_include_pad = 1` + ceil_mode -> constant divisor includes the overhang cells, wrong.

## Reproducer (1D)

Input `[1,2,3,4,5,6]`, kernel 3, stride 2, no padding, `ceil_mode=1`. Three windows; the third
covers input indices [4,5] (index 6 is the overhang).

- Expected (PyTorch `avg_pool1d`, count_include_pad=False, same as the ONNX definition):
  `[2.0, 4.0, 5.5]`  ((5+6)/2 = 5.5)
- SOFIE generated code: `[2.0, 4.0, 3.6667]`  ((5+6)/3 = 3.6667)

The first two windows are full so they match; only the overhung window is wrong.

## Class

Same pattern as the Conv stride/dilation and Pool asymmetric-padding bugs: a divisor branch the
generator emits for one configuration (`count_include_pad==0 && padding`) but not for the partial
window that `ceil_mode` produces, so the wrong path stays hidden until a `ceil_mode` test exists.
