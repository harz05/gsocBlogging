# A2: batch the non-grouped Conv GEMM with gemmStridedBatched

- Issue: [ML4EP/SOFIE #29](https://github.com/ML4EP/SOFIE/issues/29) (OPEN)
- PR: [ML4EP/SOFIE #30](https://github.com/ML4EP/SOFIE/pull/30) (OPEN), +164 -22
- Branch: `feat/conv-batched-gemm` off `gpu/alpaka`
- Files: `core/inc/SOFIE/ROperator_Conv.hxx`; uses the `gemmStridedBatched` API in
  `sofieBLAS` (commit `fa108fb`)

The main performance result so far. The non-grouped Conv path ran one `blas.matmul` per batch
sample; this replaces the loop with a single batched GEMM.

## The problem

In `Generate_GPU_ALPAKA`, the non-grouped path runs one `blas.matmul` per batch sample. Each
iteration does im2col into the shared `_xcol` buffer, broadcasts bias, calls matmul, then
`alpaka::wait(queue)` before the next sample, because the next im2col would overwrite `_xcol`
while the current GEMM is still reading it. For batch B that is B separate small GEMMs and about
`3B` sync points. Every regular `blas.matmul` also ends in a `cudaStreamSynchronize`, a full GPU
stall, so these are forced serialization points, not just launch overhead.

For every sample the weight matrix `_f` is identical; only the im2col input and the output slice
change. That is exactly what `gemmStridedBatched` handles.

## Why strided-batched and not collapse

The mentor's Gemm optimization (commit `b7e5178`) added two batched paths to `ROperator_Gemm`: a
collapse path (`batchCollapseB`, folding the batch into a matrix dimension as one big GEMM) for a
shared weight, and a `gemmStridedBatched` path (`useSBatched`) for a varying weight, with a
serial fallback. A2 extends that same loop-to-batched conversion to Conv.

For Conv, collapse is faster in raw GEMM but folds the batch into a matrix dimension, producing
`[batch, spatial, channel]` while Conv needs `[batch, channel, spatial]`. That requires a
transpose whose cost erodes the gain, worst exactly in the small-GEMM HEP regime. Strided-batched
keeps the batch as a separate outer axis (the `batchCount`), so it is layout-safe by construction
and unifies cleanly with the grouped path later. So A2 uses `gemmStridedBatched`, not collapse.

`gemmStridedBatched` (the new sofieBLAS API) is a thin wrapper around
`cublasSgemmStridedBatched`. Its key property here: it does not call `cudaStreamSynchronize`, so
the operations stay asynchronous on the stream. A stride of 0 for an operand means "broadcast,
use the same buffer for all batches", which is exactly the shared weight case.

## The change (non-grouped path only; grouped unchanged)

Four edits in `ROperator_Conv.hxx`:

1. `Initialize`: `_xcol` is sized to hold all B samples' im2col output
   (`B x colElements`) instead of one slice, so the batches can coexist instead of overwriting a
   shared buffer.
2. `GetBlasConfig`: returns empty for the non-grouped path (legacy cuBLAS, no cuBLASLt layout
   needed).
3. The non-grouped loop body now only does im2col into its own slice `_xcol + n*colElements`
   (plus bias into the output slice). No inner GEMM, no inter-sample waits.
4. After the batch loop: one `gemmStridedBatched('n','n', gemm_m, gemm_n, gemm_k, 1, _xcol,
   gemm_m, colElements, _f, gemm_k, 0, beta, _Y, gemm_m, gemm_n*gemm_m, bsize)`, with `beta = 1`
   if there is bias else 0. `strideB = 0` broadcasts the shared weight.

Sync count for a non-grouped conv drops from about `3B` to 2. Batch=1 also routes through
`gemmStridedBatched` with `batchCount=1`. The grouped path is unchanged (still a per-(batch,
group) matmul loop); batching the grouped path (A2b) is deferred until the grouped gemm_n fix
PR #24 lands (see [../week1/grouped-conv-regression.md](../week1/grouped-conv-regression.md)).

## The memory cost

`_xcol` must grow B times so all samples' im2col coexist:

```
extra_xcol_bytes = (B - 1) * (inC/G) * kH*kW * oH*oW*oD * sizeof(float)
```

For HEP-scale models (small spatial, modest channels) this is trivial. For ResNet-class image
conv at large batch it is large (a ResNet-18 first conv at batch 32 is on the order of hundreds
of MB), which is the case where this approach would need a fallback or be scoped out.

## Benchmark results (Colab T4)

The fork's benchmark tool constructs the session outside the timed loop. A/B by reverting
`Conv.hxx` to the base commit. Speedup is baseline (per-sample loop) divided by A2.

8-layer conv stack, C=16, 16x16 spatial, warmup 20, 1000 timed iters:

| Batch | Baseline (ms) | A2 (ms) | Speedup |
|---|---|---|---|
| 1 | ~0.44 | ~0.40 | ~1.0x (wash) |
| 4 | 1.24 | 0.47 | 2.6x |
| 8 | 2.30 | 0.68 | 3.4x |
| 16 | 4.54 | 1.11 | 4.1x |

The baseline scales about linearly in B (B serial GEMMs, three syncs each). A2 is sub-linear
(one batched GEMM plus a fixed two syncs), so it turns O(B) overhead into O(1).

Crossover study (single conv layer, to find where the win shrinks as the GEMM grows). Batch
sweep at C32, 16x16: B1 1.0x, B8 2.4x, B32 3.8x. Size sweep at B8 (GEMM grows, win shrinks):
C16/8x8 2.6x, C32/16x16 2.4x, C64/32x32 1.9x, C64/56x56 1.33x, C128/28x28 1.17x. The speedup
decreases monotonically with GEMM size, and GEMM shape matters too (a squarer C128 drops more
than a tall-skinny C64/56 at similar FLOPs). It never regresses (at least 1.17x even at the
largest tested) and is exactly neutral at batch=1.

Honest claim: 2 to 4x for batched small and medium-GEMM conv (the HEP regime), tapering to about
1.1 to 1.3x for large GEMMs, neutral at batch=1, no regression. Reaching a true 1.0x would need
ResNet conv1 (224x224) scale.

## Status notes

Issue #29 frames the improvement; PR #30 implements the non-grouped path. The grouped batched
path (A2b) and a ResNet-scale large-GEMM measurement are the open follow-ups. Build and benchmark
commands are in [../BUILD-NOTES.md](../BUILD-NOTES.md).
