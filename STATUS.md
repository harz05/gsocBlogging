# Status of all issues and PRs

The single index of everything raised and filed across the two repos. Status verified live via
`gh` on 2026-06-04. Re-run the commands at the bottom to refresh.

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
| Issue #22473 | Conv generates invalid code and crashes for dilation > 1 | OPEN | https://github.com/root-project/root/issues/22473 |
| PR #22474 | Fix double-counted dilation in Conv im2col + regression test | OPEN | https://github.com/root-project/root/pull/22474 |

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

PR #20 closes the earlier batch-test issue #18; #18 itself is referenced for context but is not
tracked separately here.

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

## Tally as of 2026-06-04

- 17 items tracked: 6 on upstream ROOT (3 issues, 3 PRs), 11 on the fork (5 issues, 6 PRs).
- Merged: 2 ROOT PRs (#21730, #22438). Closed issues: 2 ROOT (#21729 fixed by #21730,
  #22436 fixed by #22438).
- Open: ROOT issue #22473 + PR #22474; fork issues #21, #23, #25, #26, #29 and PRs #20, #22,
  #24, #27, #28, #30.

## Refresh commands

```bash
# fork items
for n in 21 23 25 26 29; do gh issue view $n --repo ML4EP/SOFIE --json number,title,state; done
for n in 20 22 24 27 28 30; do gh pr view $n --repo ML4EP/SOFIE --json number,title,state,mergedAt; done
# upstream items
for n in 21729 22436 22473; do gh issue view $n --repo root-project/root --json number,title,state; done
for n in 21730 22438 22474; do gh pr view $n --repo root-project/root --json number,title,state,mergedAt; done
```
