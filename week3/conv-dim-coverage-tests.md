# Conv GPU coverage: bias, 1D and 3D (PR #31)

Fork PR: https://github.com/ML4EP/SOFIE/pull/31 (branch `feat/conv-dim-coverage-tests`)

After the batch (#20) and groups (#22) tests, three more Conv GPU code paths were still
emitted by the generator but never exercised by any alpaka test. I added a GTest for each.

## The three untested paths

1. **Bias** - the `BiasBroadcastKernel` plus the GEMM beta=1 accumulate. Every existing Conv
   alpaka model had no bias input, so that kernel never ran. Model `ConvWithBias`: x[1,1,5,5]
   iota, W[2,1,3,3] ones, bias [10, 100]. The two distinct biases make a wrong-channel
   broadcast visible.
2. **Conv1D** (`fDim==1`) - the 1D branches in the weight-vec and im2col kernels. All other
   Conv alpaka models are 2D. Model `Conv1d`: x[1,2,7], W[3,2,3] iota weights, bias [1,2,3].
3. **Conv3D** (`fDim>2`) - the depth branches and depth-padding handling in im2col. Model
   `Conv3d`: x[1,1,3,4,4], W[2,1,2,3,3] iota, bias [5,50], pads h/w by 1, no depth pad.

Iota weights give each output and input channel a distinct slice so a wrong index shows up.
W and bias are ONNX initializers (baked into the .dat), so `infer()` takes only x.

## Verification

Each reference was computed with PyTorch (`F.conv1d` / `F.conv2d` / `F.conv3d`, cross-correlation,
no kernel flip) reading W and bias straight from the committed .onnx, and matched within 1e-3.
On Colab T4 all three pass and the existing Conv tests still pass.

The build succeeding already cleared the codegen-crash risk for 1D and 3D: the test binary
includes the generated `_FromONNX_GPU_ALPAKA.hxx`, so if `GenerateGPU_ALPAKA` had thrown for
those dimensions the build would have failed before any test ran.

## Note

Generators were not committed (the build globs `*.onnx`; matches the #20/#22 convention). The
three onnx models, the three references, and the wired `TEST_F` cases are the whole PR.
