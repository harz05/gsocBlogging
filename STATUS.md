# Status of all issues and PRs

The single index of everything raised and filed across the two repos. Status verified live via
`gh` on 2026-06-10. Re-run the commands at the bottom to refresh.

Two repos are in play:
- **root-project/root** is upstream ROOT. SOFIE's CPU code generation lives here in
  `tmva/sofie/`. Fixes here help every SOFIE user, GPU or not.
- **ML4EP/SOFIE** is the fork where the alpaka GPU backend is developed, on the `gpu/alpaka`
  branch. All GSoC GPU work targets this fork.

## Upstream ROOT (root-project/root, CPU SOFIE)

| Item | Title | State | Link |
|---|---|---|---|
| Issue #21729 | Conv dynamic-shape output formula generates wrong C++ when stride > 1 | CLOSED 2026-05-03 | https://github.com/root-project/root/issues/21729 |
| PR #21730 | Fix Conv dynamic-shape output formula when stride > 1 | MERGED 2026-05-03 | https://github.com/root-project/root/pull/21730 |
| Issue #22436 | MaxPool/AvgPool generate wrong code for asymmetric padding | CLOSED 2026-05-31 | https://github.com/root-project/root/issues/22436 |
| PR #22438 | Fix MaxPool/AvgPool pad indices for asymmetric padding | MERGED 2026-05-31 | https://github.com/root-project/root/pull/22438 |
| Issue #22473 | Conv generates invalid code and crashes for dilation > 1 | CLOSED 2026-06-05 | https://github.com/root-project/root/issues/22473 |
| PR #22474 | Fix double-counted dilation in Conv im2col + regression test | MERGED 2026-06-05 | https://github.com/root-project/root/pull/22474 |
| Issue #22523 | AveragePool with ceil_mode divides the partial/overhang window by the full kernel size | OPEN | https://github.com/root-project/root/issues/22523 |
| Issue #22552 | Register the existing Swish operator in the ONNX parser | OPEN | https://github.com/root-project/root/issues/22552 |

## Fork (ML4EP/SOFIE, base branch `gpu/alpaka`)

| Item | Title | State | Link |
|---|---|---|---|
| Issue #21 | Add GTests for Conv GPU inference with groups > 1 | OPEN | https://github.com/ML4EP/SOFIE/issues/21 |
| PR #20 | GTests for Conv GPU batch>1 inference | OPEN | https://github.com/ML4EP/SOFIE/pull/20 |
| PR #22 | GTests for Conv GPU groups>1 inference | OPEN | https://github.com/ML4EP/SOFIE/pull/22 |
| Issue #23 | Incorrect output from grouped Conv GPU path | OPEN | https://github.com/ML4EP/SOFIE/issues/23 |
| PR #24 | Fix for incorrect output from grouped Conv (gemm_n / fAttrGroup) | OPEN | https://github.com/ML4EP/SOFIE/pull/24 |
| Issue #25 | Move WeightVecKernel out of infer() into session init (A1) | OPEN (no PR yet) | https://github.com/ML4EP/SOFIE/issues/25 |
| Issue #26 | MaxPool has no GPU implementation | OPEN | https://github.com/ML4EP/SOFIE/issues/26 |
| PR #27 | Pool (MaxPool, AvgPool, GlobalAveragePool) GPU support + tests | OPEN | https://github.com/ML4EP/SOFIE/pull/27 |
| PR #28 | Fix wrong pad indices in Pool for asym padding (port of root #22438) | OPEN | https://github.com/ML4EP/SOFIE/pull/28 |
| Issue #29 | Batch the non-grouped Conv GEMM with gemmStridedBatched (A2) | OPEN | https://github.com/ML4EP/SOFIE/issues/29 |
| PR #30 | Batch the non-grouped Conv GEMM with gemmStridedBatched | OPEN | https://github.com/ML4EP/SOFIE/pull/30 |
| PR #31 | GTests for Conv GPU bias, 1D and 3D paths | OPEN | https://github.com/ML4EP/SOFIE/pull/31 |
| Issue #32 | Conv generates wrong code for dilation > 1 (GPU silent-wrong, CPU segfault) | OPEN | https://github.com/ML4EP/SOFIE/issues/32 |
| Issue #33 | Add GTest for Conv GPU SAME_UPPER autopad | OPEN | https://github.com/ML4EP/SOFIE/issues/33 |
| PR #34 | Fix dilation>1 in Conv operator (GPU & CPU) | OPEN | https://github.com/ML4EP/SOFIE/pull/34 |
| PR #35 | Add GTest for Conv GPU SAME_UPPER autopad | OPEN | https://github.com/ML4EP/SOFIE/pull/35 |

PR #20 closes the earlier batch-test issue #18; #18 itself is referenced for context but is not
tracked separately here. PR #34 closes #32; PR #35 closes #33.

## Where each item is written up

- Issue #21729 / PR #21730 -> [community-bonding/first-root-pr-conv-dynshape.md](community-bonding/first-root-pr-conv-dynshape.md)
- PR #20, Issue #21 -> [community-bonding/conv-gpu-untested-paths.md](community-bonding/conv-gpu-untested-paths.md)
- PR #22 -> [week1/conv-group-tests.md](week1/conv-group-tests.md)
- Issue #23 / PR #24 -> [week1/grouped-conv-regression.md](week1/grouped-conv-regression.md)
- Issue #25 (proposal) -> [week1/weightvec-init-optim.md](week1/weightvec-init-optim.md); implementation -> [week2/weightvec-init-A1-impl.md](week2/weightvec-init-A1-impl.md)
- Issue #26 / PR #27 -> [week1/maxpool-gpu.md](week1/maxpool-gpu.md); AvgPool/GlobalAvgPool extension -> [week2/avgpool-globalavgpool-gpu.md](week2/avgpool-globalavgpool-gpu.md)
- Issue #22436 / PR #22438 -> [week1/root-pool-asym-pad-bug.md](week1/root-pool-asym-pad-bug.md)
- PR #28 -> [week2/pool-padfix-port.md](week2/pool-padfix-port.md)
- Issue #22473 / PR #22474 -> [week2/conv-dilation-bug.md](week2/conv-dilation-bug.md)
- Issue #29 / PR #30 -> [week2/conv-batched-gemm.md](week2/conv-batched-gemm.md)
- PR #31 -> [week3/conv-dim-coverage-tests.md](week3/conv-dim-coverage-tests.md)
- Issue #32 / PR #34 -> [week3/conv-dilation-fix.md](week3/conv-dilation-fix.md)
- Issue #33 / PR #35 -> [week3/conv-autopad-upper-test.md](week3/conv-autopad-upper-test.md)
- Issue #22523 -> [week3/avgpool-ceilmode-bug.md](week3/avgpool-ceilmode-bug.md)
- Issue #22552 -> [week3/swish-parser-gap.md](week3/swish-parser-gap.md)

## Tally as of 2026-06-10

- 24 items tracked: 8 on upstream ROOT (5 issues, 3 PRs), 16 on the fork (7 issues, 9 PRs).
- Merged: 3 ROOT PRs (#21730, #22438, #22474). Closed issues: 3 ROOT (#21729 by #21730,
  #22436 by #22438, #22473 by #22474).
- Open: ROOT issues #22523, #22552; all 16 fork items (issues #21, #23, #25, #26, #29, #32, #33
  and PRs #20, #22, #24, #27, #28, #30, #31, #34, #35).

## Refresh commands

```bash
# fork items
for n in 21 23 25 26 29 32 33; do gh issue view $n --repo ML4EP/SOFIE --json number,title,state; done
for n in 20 22 24 27 28 30 31 34 35; do gh pr view $n --repo ML4EP/SOFIE --json number,title,state,mergedAt; done
# upstream items
for n in 21729 22436 22473 22523 22552; do gh issue view $n --repo root-project/root --json number,title,state; done
for n in 21730 22438 22474; do gh pr view $n --repo root-project/root --json number,title,state,mergedAt; done
```
