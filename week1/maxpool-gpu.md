# MaxPool GPU (1D, 2D, 3D)

- Issue: [ML4EP/SOFIE #26](https://github.com/ML4EP/SOFIE/issues/26) (OPEN)
- PR: [ML4EP/SOFIE #27](https://github.com/ML4EP/SOFIE/pull/27) (OPEN), closes #26, +293 -1
- File: `core/inc/SOFIE/ROperator_Pool.hxx`

The first GPU operator added beyond what already existed on `gpu/alpaka`. Until this PR,
`ROperator_Pool` did not override any of the `Generate_GPU_*` methods, so the alpaka backend
fell through to the base class and silently emitted empty kernels for MaxPool nodes. The CPU
side had the full implementation (1D, 2D, 3D, max and average pool); the GPU side had nothing.

## What the PR adds

GPU code generation for MaxPool 1D, 2D, and 3D via three `Generate_GPU_*` methods on
`ROperator_Pool`. The kernels are per-operator constexpr alpaka functors, one thread per output
element, with a row-major index-decode formula. The window reduction math is the same as the CPU
`Generate()`. `OperatorKind::POOL` is added to the enum and wired into the constructor.

## Scope

This is a baseline implementation: one thread per output element, attributes baked in as
constexpr. It is correct and gives a working GPU MaxPool that later optimizations (shared-memory
tiling, fusion) can build on. AveragePool and GlobalAveragePool follow the same indexing pattern
and are the natural next operators.

## Follow-up

The asymmetric-padding pad-index bug fixed upstream in root #22438 (see
[root-pool-asym-pad-bug.md](root-pool-asym-pad-bug.md)) also affects the fork's Pool operator,
both the CPU `Generate()` and this new GPU kernel. The CPU port is PR #28 (see
[../week2/pool-padfix-port.md](../week2/pool-padfix-port.md)); the GPU kernel added here needs the
same pad-index correction as a follow-up.
