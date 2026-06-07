# Grouped Conv GPU regression: gemm_n never divided by group count

- Issue: [ML4EP/SOFIE #23](https://github.com/ML4EP/SOFIE/issues/23) (OPEN)
- PR: [ML4EP/SOFIE #24](https://github.com/ML4EP/SOFIE/pull/24) (OPEN), closes #23, +2 -1
- File: `core/inc/SOFIE/ROperator_Conv.hxx`

This is the bug that the groups>1 tests ([conv-group-tests.md](conv-group-tests.md)) caught. It
is the same class as the dynamic-shape and pool-pad bugs: an emitted code path that no test had
run until now.

## The failure fingerprint

On the restructured `gpu/alpaka` HEAD, with the same ONNX, the same reference, and the same
TEST_F code, the grouped tests failed in a very specific way. For ConvGroup2: output indices
`i=0..49` passed and `i=50..99` failed with large errors. The 50/50 split corresponds exactly
to "output channels 0 and 1 correct, channels 2 and 3 wrong." That pattern only fits one cause:
each per-group matmul was computing all output channels using group 0's input, and the second
group's matmul wrote nothing useful.

## Tracing it

Comparing the old Conv code (where the tests passed) with the new (where they failed), the
grouped emission code itself was unchanged. The difference was in setup. The old code divided
`gemm_n` by the group count:

```cpp
gemm_n = fShapeW[0];          // total output channels
if (fAttrGroup > 1) {
    gemm_n /= fAttrGroup;     // per-group output channels
}
```

The new code dropped the division during an operator-init refactor:

```cpp
size_t gemm_n = outChannels;  // total output channels
// comment claimed "we divide per group at launch" -- but no launch site divides
```

So for groups=2 with outC=4, every per-group matmul ran with `n = 4` (total) instead of `n = 2`
(per group). Each per-group matmul computed all four output channels from group 0's input.

## The fix

Two places, both adding the missing division:

- In `Generate_GPU_ALPAKA`, after `gemm_n = outChannels`, add
  `if (fAttrGroup > 1) gemm_n /= fAttrGroup;`.
- In `GetBlasConfig`, the same fix for `gemm_n_`.

The second place is the subtle one. The first fix attempt changed only `gemm_n` in the matmul
caller, and the tests still failed. `GetBlasConfig` independently computes `gemm_n_` from
`fShapeW[0]` (total) and feeds it to `addLayoutConfig`, which registers a cuBLASLt layout keyed
on `(m, n)`. After the caller-side fix, the matmul asked for the per-group `n`, a key that was
never registered, and the layout lookup threw `out_of_range`. Both the setup function and the
usage function compute the same dimension, so both have to agree.

After the fix all tests pass, including the three new ConvGroup tests.

## Lesson

When a setup function and a usage function each compute the same dimension independently, a
refactor that updates one side and not the other produces exactly this kind of mismatch. The
division likely did not carry over from the old `Initialize()` during the operator-init
refactor.
