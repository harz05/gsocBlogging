# Swish operator not registered in the ONNX parser (ROOT #22552)

ROOT issue: https://github.com/root-project/root/issues/22552 (improvement)

Found while studying `tmva/sofie_parsers/` for gaps of the type "an operator the codegen supports
but the parser cannot reach".

## The gap

SOFIE has a complete `ROperator_Swish` that computes `x / (1 + exp(-x))` = `x*sigmoid(x)`. It has
the full method set (`Initialize`, `ShapeInference`, `TypeInference`, `Generate`, `GetStdLibs`),
is structurally identical to `ROperator_Sigmoid`, and is listed in `OperatorList.hxx`.

But the ONNX parser never registers `"Swish"`. ONNX has had a standard `Swish` op since opset 24
(`Swish(x) = x*sigmoid(alpha*x)`, alpha default 1). So a model containing a Swish node fails to
parse with `Operator type Swish is not yet supported`, even though the operator is implemented.

The only difference between Swish and Sigmoid is that Sigmoid has a `ParseSigmoid.cxx` and a
`RegisterOperator("Sigmoid", ...)` call, and Swish has neither.

## How it was verified

- Diffed every defined parser against every registered one: registration is otherwise clean, no
  dead parsers. Swish is the one operator with codegen but no parser.
- Checked the actual installed ONNX schema registry: `Swish` exists since opset 24, default domain,
  one input X, one attribute `alpha` default 1.0.
- A `ReduceMax`/`ReduceMin` lead turned out to be a false alarm: the `EReduceOpMode` enum has only
  Mean/Sum/SumSquare/Prod, so those are simply not implemented (a feature, not a registration gap).

## The fix (scoped in the issue)

Add a `ParseSwish.cxx` mirroring `ParseSigmoid` plus `RegisterOperator("Swish", ParseSwish)`. The
operator implements `alpha = 1` (the default), so the parser should read the optional `alpha`
attribute and either reject `alpha != 1` or extend the operator with an alpha parameter. A minor
aside: the operator's `Generate` uses the older static-length style rather than
`ConvertDimShapeToLength`, so it would not handle dynamic shapes - optional to modernise in the
same change.

This also ties to the proposal: week-7 work includes registering Swish in the parser and adding the
GPU kernel; the parser-registration half is a standalone contribution that applies to ROOT's CPU
path independent of the GPU kernel.
