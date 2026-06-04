# Conv dynamic-shape output formula, stride > 1

- Issue: [root-project/root #21729](https://github.com/root-project/root/issues/21729) (CLOSED)
- PR: [root-project/root #21730](https://github.com/root-project/root/pull/21730) (MERGED 2026-05-03)
- File: `tmva/sofie/inc/TMVA/ROperator_Conv.hxx`

This was the first upstream contribution. It is a CPU-side SOFIE code-generation bug, not GPU,
but it sits in the same Conv operator the GPU work centers on, so reading it was a good way into
the codebase.

## The bug

When a Conv has a parameterized (dynamic) spatial dimension, SOFIE emits a C++ expression for
the output size instead of a constant. The convolution output formula is
`(input + pad - kernel) / stride + 1`. In the `computeOutput` lambda the trailing `+ 1` was
being concatenated onto the stride string rather than added as a separate term:

```cpp
int64_t v = pad - kernel;
std::string outStr = "((" + inputDim.param + "+" + std::to_string(v) + ")/"
                   + std::to_string(stride) + "1)";   // BUG: "1" glued onto the stride
```

For `stride = 2` this generates `((W+-3)/21)`, which divides by 21 instead of dividing by 2 and
then adding 1. The generated tensor-info line came out as
`dynamicTensorInfos.push_back( {0, 1, 4* (((W+-3)/21)) });`. The output is silently wrong; with
stride 1 the mistake is invisible because `/11` happens not to be exercised the same way and
every existing test used stride 1 on dynamic dims.

## Reproducer

Build a Conv ONNX model with a dynamic spatial dim and stride 2, parse it, generate, and read
the emitted expression:

```python
import numpy as np, onnx
from onnx import TensorProto, helper, numpy_helper
W = numpy_helper.from_array(np.ones((1,1,3), dtype='float32'), name='W')
X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 1, 'W'])
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 1, 'out_W'])
node = helper.make_node('Conv', ['X','W'], ['Y'], kernel_shape=[3], strides=[2], pads=[0,0])
graph = helper.make_graph([node], 'test', [X], [Y], initializer=[W])
onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid('',13)]), '/tmp/conv_dyn.onnx')
```

```python
import ROOT
m = ROOT.TMVA.Experimental.SOFIE.RModelParser_ONNX().Parse('/tmp/conv_dyn.onnx')
m.Generate(); m.OutputGenerated('/tmp/conv_dyn.hxx')
# grep the generated header: the mangled "/21" shows the bug
```

## The fix

Correct the emitted string so the formula reads `(... )/stride + 1`:

```cpp
std::string outStr = "((" + inputDim.param + "+" + std::to_string(v) + ")/"
                   + std::to_string(stride) + "+1)";
```

The PR adds the regression test `ConvWithDynShapeStride` plus its ONNX model
(`tmva/sofie/test/input_models/ConvWithDynShapeStride.onnx`) and the test case in
`TestCustomModelsFromONNX.cxx`.

## Why it matters

It is the simplest member of the recurring theme: an emitted code path (dynamic-shape output,
stride > 1) that no test covered. The fix is one character, the value was writing the first test
that exercises that path. See [../BUILD-NOTES.md](../BUILD-NOTES.md) for the local build and the
`ctest -R tmva-sofie-test-TestCustomModelsFromONNX` workflow used to verify it.
