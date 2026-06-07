# A1: move WeightVecKernel out of infer() into session init

- Issue: [ML4EP/SOFIE #25](https://github.com/ML4EP/SOFIE/issues/25) (OPEN, no PR yet)
- File: `core/inc/SOFIE/ROperator_Conv.hxx`, hook in `core/inc/SOFIE/ROperator.hxx`

Proposed in Week 1 while reading Conv.hxx for optimization opportunities. The implementation and
on-GPU verification land in Week 2 (see
[../week2/weightvec-init-A1-impl.md](../week2/weightvec-init-A1-impl.md)). This file is the
proposal and the reasoning; that file is the result.

## The observation

`Generate_GPU_ALPAKA` emits the `WeightVecKernel` launch inside the `_infer_impl` body. That
kernel reorders the weight tensor `W` into the dilated layout `_f` that the GEMM consumes. The
weights are constant after they are loaded from the `.dat` file in the session constructor. So
this reordering runs on every `infer()` call when it only needs to run once, at session init.

## The hook already exists

The base operator already exposes `GenerateInitCode_GPU_ALPAKA()` in `ROperator.hxx`, called
once per operator from inside the session constructor body in `RModel_ALPAKA.cxx`, after the
weights have been uploaded to `deviceBuf_W`. Most operators return an empty string. Conv does
not override it. So the right hook for one-time, post-weight-upload init code already exists;
Conv simply does not use it.

## Expected win

One kernel launch and one `alpaka::wait(queue)` skipped per inference call, on the order of 10
to 70 microseconds. That is roughly 1 to 10 percent at batch=1 (the latency-sensitive HEP
trigger case) and smaller at larger batch. It is not the big optimization (that is A2, see
[../week2/conv-batched-gemm.md](../week2/conv-batched-gemm.md)). Output is bit-identical because
the weights are immutable after session init, so there is no numerical risk.
