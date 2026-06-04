# Community bonding (through 2026-05-24)

The window between selection and the start of the coding period. Coding officially starts
2026-05-25. This period was about getting set up to contribute to the GPU backend: building
SOFIE in both repos, reading the alpaka code generation path end to end, and exercising parts
of the Conv GPU path that existed in code but had never been run.

## What the project is

The GSoC project is "ML Inference on Heterogeneous Architectures using SOFIE" under
CERN-HSF / ML4EP / ROOT, mentored by Lorenzo Moneta and Sanjiban Sengupta. SOFIE is ROOT's
tool that turns a trained ONNX model into a standalone C++ header so inference can run inside
C++ frameworks with only BLAS as a runtime dependency. The alpaka backend extends that
generated-code approach to GPUs: one kernel implementation compiles to NVIDIA CUDA, AMD HIP,
and CPU without changes.

At the time of the ACAT 2025 presentation the alpaka path could run activations, element-wise
ops, and single-sample Conv2D, but a full CNN (Conv + BatchNorm + Pool + Linear) could not run
entirely on GPU. The project closes that gap. Slide 77 of the ACAT deck shows ResNet18
benchmarked via the SYCL backend but not alpaka, because alpaka could not yet run Conv batch>1,
BatchNorm, and Pool. Closing that is the headline target.

## A theme that runs through all of this work

Most of the bugs found here and in later weeks are the same class: a code path that the
generator emits but that no test ever exercises, so a latent bug sits hidden until someone
writes the first test for it. Every existing Conv test used input shape `[1,1,H,W]` (batch 1,
group 1) and zero padding, so the batch loop, the group loop, the asymmetric-pad path, and the
dilation path all went unverified. Writing the first test for each is what surfaced the bugs.
This is worth keeping in mind: the highest-value contribution was often not new code, it was
being the first to actually run an existing path.

## Items in this period

- [first-root-pr-conv-dynshape.md](first-root-pr-conv-dynshape.md) — ROOT issue #21729 and
  PR #21730, the first upstream contribution, merged 2026-05-03. A string-concatenation bug in
  Conv dynamic-shape output code generation.
- [conv-gpu-untested-paths.md](conv-gpu-untested-paths.md) — fork PR #20 (Conv GPU batch>1
  tests) and issue #21 (Conv GPU groups>1 tests). Identifying and starting to cover the
  emitted-but-untested Conv GPU paths.

See [../STATUS.md](../STATUS.md) for the live status of every item and
[../BUILD-NOTES.md](../BUILD-NOTES.md) for how to build and test in both repos.
