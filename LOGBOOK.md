# Logbook

Raw running record of what I did and found, newest entries appended at the bottom. This is my
own reference stream. The polished per-week writeups distil from this; see the week folders and
[STATUS.md](STATUS.md) for the structured version.

Two repos:
- ROOT = root-project/root (upstream, CPU SOFIE in `tmva/sofie/`)
- fork = ML4EP/SOFIE (alpaka GPU backend, branch `gpu/alpaka`)

---

## Community bonding (through 2026-05-24)

- Built SOFIE in both repos, read the alpaka code-generation path end to end. Build traps and
  the working commands are in [BUILD-NOTES.md](BUILD-NOTES.md).
- 2026-03-28: filed ROOT issue #21729, Conv dynamic-shape output formula generates wrong C++ for
  stride > 1 (the `+1` glued onto the stride string, `((W+-3)/21)`). Opened PR #21730.
- 2026-05-03: ROOT PR #21730 merged. First upstream contribution.
  -> [community-bonding/first-root-pr-conv-dynshape.md](community-bonding/first-root-pr-conv-dynshape.md)
- Audited the Conv GPU path. Found two emitted-but-untested branches: batch > 1 and groups > 1.
- 2026-05-12: opened fork PR #20, tests for Conv GPU batch > 1 (ConvBatch2/4/8). Pass on T4.
- 2026-05-24: filed fork issue #21, tests needed for Conv GPU groups > 1.
  -> [community-bonding/conv-gpu-untested-paths.md](community-bonding/conv-gpu-untested-paths.md)

## Week 1 (2026-05-25 to 05-31)

- 2026-05-25: opened fork PR #22, the groups > 1 tests (ConvGroup2, ConvGroup4, ConvBatch4Group2,
  iota weights so a wrong slice is visible). On the restructured `gpu/alpaka` HEAD they FAILED:
  output channels 0-1 correct, 2-3 wrong.
- 2026-05-25: diagnosed the failure to `gemm_n` never being divided by `fAttrGroup` (dropped
  during an operator-init refactor; the "divide per group at launch" comment was kept but no
  launch divides). Filed fork issue #23, opened PR #24 (fix in both `Generate_GPU_ALPAKA` and
  `GetBlasConfig`, the second needed or the cuBLASLt layout lookup throws).
  -> [week1/conv-group-tests.md](week1/conv-group-tests.md),
     [week1/grouped-conv-regression.md](week1/grouped-conv-regression.md)
- 2026-05-26: filed fork issue #25 (A1), move the Conv weight-vec kernel out of `infer()` into
  session init using the existing `GenerateInitCode_GPU_ALPAKA` hook.
  -> [week1/weightvec-init-optim.md](week1/weightvec-init-optim.md)
- 2026-05-29: filed fork issue #26 (MaxPool has no GPU impl) and opened PR #27, MaxPool 1D/2D/3D
  GPU kernels (one thread per output, constexpr attrs).
  -> [week1/maxpool-gpu.md](week1/maxpool-gpu.md)
- 2026-05-30: found the ROOT CPU pool asymmetric-padding bug (end pads read at interleaved
  indices 1/3/5 instead of grouped fDim/fDim+1/fDim+2). Filed ROOT issue #22436.
- 2026-05-31: ROOT PR #22438 merged (pool pad fix + MaxPool2d_AsymPad regression test).
  -> [week1/root-pool-asym-pad-bug.md](week1/root-pool-asym-pad-bug.md)

## Week 2 (2026-06-01 to 06-07, in progress)

- 2026-06-01: opened fork PR #28, porting the pool pad fix into the fork's CPU Pool operator.
  GPU MaxPool kernel (#27) still needs the same correction as a follow-up.
  -> [week2/pool-padfix-port.md](week2/pool-padfix-port.md)
- 2026-06-02: implemented A1 (issue #25) on branch `feat/conv-weightvec-init`. Chose the
  signature-change option (hook takes opName via `std::to_string(id)`, matching the sibling
  GPU-gen methods). Verified bit-identical on Colab T4, all six Conv tests pass, weight-vec block
  confirmed relocated into the Session constructor.
  -> [week2/weightvec-init-A1-impl.md](week2/weightvec-init-A1-impl.md)
- 2026-06-03: found the ROOT Conv dilation > 1 crash (dilation double-counted: inflated
  `fAttrKernelShape` AND `fAttrDilations` both passed to im2col, OOB -> segfault). Filed ROOT
  issue #22473, opened PR #22474 (one-line fix: set `fAttrDilations` to 1 after the weight
  reorder, plus a ConvWithDilation regression test). ctest went 67% -> 100%.
  -> [week2/conv-dilation-bug.md](week2/conv-dilation-bug.md)
- 2026-06-03: implemented A2 (issue #29), opened PR #30 on branch `feat/conv-batched-gemm`.
  Replaced the per-sample matmul loop with one `gemmStridedBatched` (strideB=0 broadcast weight),
  `_xcol` resized to B slices, sync count `3B` -> 2. Verified bit-identical (ConvBatch4 + the six
  batch=1 tests). Benchmarked on T4: 2.6x at B4, 3.4x at B8, 4.1x at B16 on an 8-layer C16 stack;
  neutral at B1; crossover study shows the win tapers to ~1.1-1.3x as the GEMM grows, never
  regresses. Grouped batched path (A2b) deferred until #24 merges.
  -> [week2/conv-batched-gemm.md](week2/conv-batched-gemm.md)
- 2026-06-07: extended PR #27 to AveragePool and GlobalAveragePool GPU. The MaxPool kernel
  methods now also emit an AvgPool kernel: same one-thread-per-output index math, the reduction
  switched from max to sum plus a divide. The divisor follows the CPU operator exactly, the
  in-bounds cell count at run time when count_include_pad is 0 with padding, otherwise the
  constant kernel area. GlobalAveragePool needs no separate kernel, it reaches the GPU path as an
  AveragePool with kernel equal to the image size. Added four GTests: AvgPool reusing the existing
  model and its CPU reference, AvgPoolPad for the run-time count path, AvgPoolCountIncludePad, and
  GlobalAvgPool2d. All 89 alpaka tests pass on Colab T4. PR retitled to cover all pool modes.
  -> [week2/avgpool-globalavgpool-gpu.md](week2/avgpool-globalavgpool-gpu.md)

## Week 3 (2026-06-08 to 06-14, in progress)

- 2026-06-08: opened fork PR #31, GTests for three more untested Conv GPU paths: bias
  (`BiasBroadcastKernel`), Conv1D (`fDim==1`), Conv3D (`fDim>2`). Iota weights, references
  verified against PyTorch, all pass on Colab T4.
  -> [week3/conv-dim-coverage-tests.md](week3/conv-dim-coverage-tests.md)
- 2026-06-08: filed fork issue #32, Conv dilation>1 generates wrong code on the fork too - GPU
  silently wrong (double-counted dilation between the dilated `_f` layout and the im2col), CPU
  segfault (same root cause as root #22473). And filed fork issue #33, SAME_UPPER autopad has no
  GPU test.
- 2026-06-08: filed root issue #22523, AveragePool with `ceil_mode=1` divides the overhang window
  by the full kernel size instead of the count of real cells. 1D reproducer: (5+6)/3 instead of
  (5+6)/2. -> [week3/avgpool-ceilmode-bug.md](week3/avgpool-ceilmode-bug.md)
- 2026-06-09: opened fork PR #34 fixing issue #32. CPU is a one-line port of root #22474; GPU
  needed three changes (im2col decode over the effective kernel, gather without re-applied
  dilation, and `alpaka::memset` of `_f` since alpaka buffers are not zeroed and the dilated
  layout has holes). Verified on Colab with a red->green toggle (revert only the codegen and
  `ConvWithDilation` fails). -> [week3/conv-dilation-fix.md](week3/conv-dilation-fix.md)
- 2026-06-09: opened fork PR #35 fixing issue #33. Used an odd-padding model (input 4, kernel 3,
  stride 2) instead of porting ROOT's even-padding one, because SAME_UPPER and SAME_LOWER are
  identical for even total padding. Confirmed the generated im2col uses begin pad 0 (end-heavy).
  -> [week3/conv-autopad-upper-test.md](week3/conv-autopad-upper-test.md)
- 2026-06-09: studied `tmva/sofie_parsers/`, found `ROperator_Swish` is fully implemented but
  never registered in the ONNX parser, so standard ONNX Swish (opset 24) is unparseable. Filed
  root issue #22552 (improvement). -> [week3/swish-parser-gap.md](week3/swish-parser-gap.md)

---

## How to append

Add new entries at the bottom under the current week heading, newest last. When a PR merges or a
status changes, update [STATUS.md](STATUS.md) (it has a refresh command block) rather than
editing past logbook entries; the logbook is a record of what happened when.
