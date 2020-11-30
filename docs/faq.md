Frequently Asked Questions
==========================

_Just jump to the sections that you are interested in.
Please help to raise issues if any of the document here is wrong._


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
[Full integer quantization](https://www.tensorflow.org/lite/performance/post_training_quantization#full_integer_quantization)
and quantization-aware training.
Or keep the TensorFlow/TFLite model in FP32 format.


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

