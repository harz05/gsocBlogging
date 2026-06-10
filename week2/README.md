# Week 2 (2026-06-01 to 06-07, in progress)

The second coding week. Two threads: finishing the Pool pad-fix work by porting it into the
fork, and the Conv optimization track (the A1 weight-vec move and the A2 batched GEMM), plus
another upstream Conv bug in the same untested-path class (dilation > 1).

## What happened

1. Ported the upstream pool asymmetric-padding fix (root #22438) into the fork's CPU Pool
   operator (PR #28).
2. Found that dilated Conv (dilation > 1) crashes in generated ROOT code because dilation is
   double-counted between the weight layout and im2col (root issue #22473, PR #22474).
3. Implemented A1: moved the Conv weight-vectorisation kernel into session init and verified
   it bit-identical on Colab T4.
4. Implemented A2: replaced the per-sample Conv matmul loop with a single `gemmStridedBatched`
   call, verified it, and benchmarked it (issue #29, PR #30). This is the headline performance
   result so far: 2 to 4x on batched small and medium GEMMs, neutral at batch=1, no regression.
5. Extended PR #27 from MaxPool to AveragePool and GlobalAveragePool GPU, reusing the same kernel
   structure with a sum-and-divide reduction and the CPU's count_include_pad divisor logic. Four
   new GTests, all 89 alpaka tests pass on Colab T4.

## Docs in this week

- [pool-padfix-port.md](pool-padfix-port.md) — PR #28, porting the root pool pad fix to the fork
- [conv-dilation-bug.md](conv-dilation-bug.md) — root issue #22473 + PR #22474, the dilation
  double-count crash
- [weightvec-init-A1-impl.md](weightvec-init-A1-impl.md) — A1 implemented and verified (issue
  #25, see also [../week1/weightvec-init-optim.md](../week1/weightvec-init-optim.md))
- [conv-batched-gemm.md](conv-batched-gemm.md) — issue #29 + PR #30, A2 and the benchmark study
- [avgpool-globalavgpool-gpu.md](avgpool-globalavgpool-gpu.md) — PR #27 extended to AvgPool and
  GlobalAveragePool GPU

See [../STATUS.md](../STATUS.md) for live status and [../BUILD-NOTES.md](../BUILD-NOTES.md) for
build, test, and benchmark workflows.
