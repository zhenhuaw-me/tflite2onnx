Frequently Asked Questions
==========================

_Just jump to the sections that you are interested in.
Please help to raise issues if any of the document here is wrong._


## Unsupported TFLite OP Error

As of today, `tflite2onnx` supports about 31 TFLite operators (check
the `tflite2onnx.getSupportedOperators()` API). However, there are
127 builtin operators (check via `tflite.BUILTIN_OPCODE2NAME`) in TFLite.
For the operators that is unsupported, an error like below will be thrown

```
NotImplementedError: Unsupported TFLite OP: 123 {OPNAME}
```

Usually, we need to enable that operator in `tflite2onnx`.
Please [report and contribute](contribution-guide.md)!

However, sometimes, there are operators that are not _TFLite builtin operators_
in the original model. For example, an TensorFlow operator is added
when converting TensorFlow model to TFLite like below.

```py
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                       tf.lite.OpsSet.SELECT_TF_OPS]
converter.allow_custom_ops = True
```

This is not supported currently as it requires significant end to end effort.
To workaround it, you may need to replace complicate TensorFlow operators
with [TFLite builtin operators](https://jackwish.net/tflite/docs/BuiltinOperator.m.html),
and then try again.



## FP16 Error When Converting

Related issue: [#30](https://github.com/jackwish/tflite2onnx/issues/30).

As of TensorFlow `v2.3.0`, FP16 is not natively supported by TFLite.
Operators such as [`Add`](https://github.com/tensorflow/tensorflow/blob/v2.3.0/tensorflow/lite/kernels/add.cc#L196)
and [`Conv`](https://github.com/tensorflow/tensorflow/blob/v2.3.0/tensorflow/lite/kernels/conv.cc#L998)
don't support FP16 - no related kernels.

In practice, TensorFlow inserts `tf.Cast` to converter FP16 data to FP32
for further computation. However, the MLIR based TensorFlow Lite converter
desn't support `tf.Cast`. An example of converting FP16 `tf.math.Add` is as below.

```
<unknown>:0: error: failed while converting: 'main': Ops that can be supported by the flex runtime (enabled via setting the -emit-select-tf-ops flag):
        tf.Cast {Truncate = false, device = ""}
<unknown>:0: note: see current operation: "func"() ( {
^bb0(%arg0: tensor<1x2x3x4xf16>, %arg1: tensor<1x2x3x4xf16>):  // no predecessors
  %0 = "tf.Cast"(%arg0) {Truncate = false, device = ""} : (tensor<1x2x3x4xf16>) -> tensor<1x2x3x4xf32>
  %1 = "tf.Cast"(%arg1) {Truncate = false, device = ""} : (tensor<1x2x3x4xf16>) -> tensor<1x2x3x4xf32>
  %2 = "tfl.add"(%0, %1) {fused_activation_function = "NONE"} : (tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32>) -> tensor<1x2x3x4xf32>
  "std.return"(%2) : (tensor<1x2x3x4xf32>) -> ()
}) {sym_name = "main", tf.entry_function = {control_outputs = "", inputs = "A,B", outputs = "Identity"}, type = (tensor<1x2x3x4xf16>, tensor<1x2x3x4xf16>) -> t
ensor<1x2x3x4xf32>} : () -> ()
```

In general, FP16 in a TFLite model exists due to
[FP16 quantization](https://www.tensorflow.org/lite/performance/post_training_quantization#float16_quantization).
As of today, I'd recommend to use
[full integer quantization](https://www.tensorflow.org/lite/performance/post_training_quantization#full_integer_quantization)
and quantization-aware training.
Or keep the TensorFlow/TFLite model in FP32 format.


## FP16 Quantization Model

Many people are using TFLite
[FP16 quantization](https://www.tensorflow.org/lite/performance/post_training_quantization#float16_quantization),
and some models ([example](https://github.com/jackwish/tflite2onnx/issues/33))
are published in such format.

The FP16 weights in these models will be converted to FP32 online by a TFLite
operator `Dequantize`. In general, we convert TFLite `Dequantize` to ONNX
[`DequantizeLinear`](https://github.com/onnx/onnx/blob/master/docs/Changelog.md#DequantizeLinear-10).
However, `DequantizeLinear` in ONNX supports only dequantize an integer
(`uint8`, `int8`, `int32`).

We enabled [*FP16 Quantizatoin Pattern Folding*](https://github.com/jackwish/tflite2onnx/issues/35)
to workaround this issue. In the resulted model, the FP16 tensors are converted into FP32.
Be carefull when feed or retrieve data to and from the model.

Still, I'd recommend
[full integer quantization](https://www.tensorflow.org/lite/performance/post_training_quantization#full_integer_quantization)
if possible.


## TFLite Model Contains Custom Operators

Custom operator in TFLite requires
[developer provided kernels](https://www.tensorflow.org/lite/guide/ops_custom#defining_the_kernel_in_the_tensorflow_lite_runtime).

`tflite2onnx` doesn't support custom operator as the TFLite model file
itself doesn't know how to perform the computation - which is the knowledge
of the model owner. And we don't have plan to support yet.

If your model contains custom operator, you may either break the model
into several sub-models which have no custom model, convert to ONNX
and integrate them in ONNX backend. Or you can rewrite your TensorFlow
model such that composing the custom operator with builtin operator.

I believe you have met similar in other deep learning system...
And I believe this can be resolved in the future.
But before that, we need to workaround...


## Custom the ONNX Opset Version

`tflite2onnx` is bound to ONNX Opset 11 currently.
We don't plan to support a _custom_ opset version,
since it requires opset semantic conversion
which could be a burden to handle but I don't see the value of it.

If you really need a custom opset, try
[the ONNX Version Converter](https://github.com/onnx/onnx/blob/master/docs/VersionConverter.md).
And ask them to fix if there is any bug.
(I have not used it :))
