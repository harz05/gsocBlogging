# Port the pool asymmetric-pad fix to the fork

- PR: [ML4EP/SOFIE #28](https://github.com/ML4EP/SOFIE/pull/28) (OPEN), +31 -5
- File: `core/inc/SOFIE/ROperator_Pool.hxx`
- Ports: [root-project/root #22438](https://github.com/root-project/root/pull/22438) (MERGED),
  which fixed [root #22436](https://github.com/root-project/root/issues/22436)

The upstream ROOT fix for the pool asymmetric-padding pad-index bug applies equally to the
fork's CPU Pool operator, which carries the same code. This PR ports it.

## The change

The same fix as upstream: `ROperator_Pool::Generate()` reads pads under the ONNX grouped layout
`[all_begins, all_ends]` (begins at `0, 1, 2`, ends at `fDim, fDim+1, fDim+2`) instead of the
buggy interleaved reads at indices `1, 3, 5`. The same regression test is ported across,
`MaxPool2d_AsymPad`. Full explanation of the bug and the index table is in
[../week1/root-pool-asym-pad-bug.md](../week1/root-pool-asym-pad-bug.md).

## Follow-up still needed

The GPU MaxPool kernel added in PR #27 (see [../week1/maxpool-gpu.md](../week1/maxpool-gpu.md))
has the same pad-index assumption and needs the same correction. This PR only covers the CPU
`Generate()` path; the GPU kernel fix is a follow-up once this lands.
