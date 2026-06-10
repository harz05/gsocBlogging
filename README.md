# GSoC 2026 work record: SOFIE GPU/alpaka

My running record of the GSoC project "ML Inference on Heterogeneous Architectures using SOFIE"
(CERN-HSF / ML4EP / ROOT). Documents what I did, found, and proposed, organized by week, with the
live status of every issue and PR I filed. Kept here so I have a clean reference and a base for
blog posts later.

GSoC 2026 project page: https://summerofcode.withgoogle.com/programs/2026/projects/NE74dwRu

## How this is organized

- [STATUS.md](STATUS.md) — the index: every issue and PR with live state and links.
- [BUILD-NOTES.md](BUILD-NOTES.md) — how to build and test in both repos, and the recurring
  exFAT / `.pcm` / Colab traps.
- Week folders hold the polished writeups for what happened that period:
  - [community-bonding/](community-bonding/) — through 2026-05-24
  - [week1/](week1/) — 2026-05-25 to 05-31
  - [week2/](week2/) — 2026-06-01 to 06-07
  - [week3/](week3/) — 2026-06-08 to 06-14 (in progress)

## The two repos

- **root-project/root** is upstream ROOT. SOFIE's CPU code generation lives in `tmva/sofie/`.
  Fixes here help every SOFIE user.
- **ML4EP/SOFIE** is the fork where the alpaka GPU backend is developed, on `gpu/alpaka`. The
  GPU GSoC work targets this fork.

## A recurring theme

Many of the bugs are the same class: a code path the generator emits but that no test ever runs,
so a latent bug stays hidden until the first test for it is written. Conv batch>1, Conv groups>1,
pool asymmetric padding, and Conv dilation>1 were all in this state. Writing the first test for
each is what surfaced the bug.

## Quick map of items to writeups

| Item | Where |
|---|---|
| ROOT #21729 / #21730 Conv dynshape stride | [community-bonding/first-root-pr-conv-dynshape.md](community-bonding/first-root-pr-conv-dynshape.md) |
| fork #20 batch tests, #21 group tests | [community-bonding/conv-gpu-untested-paths.md](community-bonding/conv-gpu-untested-paths.md) |
| fork #22 group tests | [week1/conv-group-tests.md](week1/conv-group-tests.md) |
| fork #23 / #24 grouped Conv regression | [week1/grouped-conv-regression.md](week1/grouped-conv-regression.md) |
| fork #25 A1 proposal | [week1/weightvec-init-optim.md](week1/weightvec-init-optim.md) |
| fork #26 / #27 MaxPool GPU | [week1/maxpool-gpu.md](week1/maxpool-gpu.md) |
| ROOT #22436 / #22438 pool asym pad | [week1/root-pool-asym-pad-bug.md](week1/root-pool-asym-pad-bug.md) |
| fork #28 pool pad fix port | [week2/pool-padfix-port.md](week2/pool-padfix-port.md) |
| ROOT #22473 / #22474 Conv dilation | [week2/conv-dilation-bug.md](week2/conv-dilation-bug.md) |
| fork #25 A1 implementation | [week2/weightvec-init-A1-impl.md](week2/weightvec-init-A1-impl.md) |
| fork #29 / #30 A2 batched GEMM | [week2/conv-batched-gemm.md](week2/conv-batched-gemm.md) |
| fork #31 Conv GPU bias / 1D / 3D tests | [week3/conv-dim-coverage-tests.md](week3/conv-dim-coverage-tests.md) |
| fork #32 / #34 Conv dilation fix (GPU + CPU) | [week3/conv-dilation-fix.md](week3/conv-dilation-fix.md) |
| fork #33 / #35 Conv SAME_UPPER autopad test | [week3/conv-autopad-upper-test.md](week3/conv-autopad-upper-test.md) |
| ROOT #22523 AveragePool ceil_mode divisor | [week3/avgpool-ceilmode-bug.md](week3/avgpool-ceilmode-bug.md) |
| ROOT #22552 Swish parser registration | [week3/swish-parser-gap.md](week3/swish-parser-gap.md) |

See [STATUS.md](STATUS.md) for the full status table. Status is current as of 2026-06-10.
