Release Notes
=============


## v0.3.0

2020-09-30, [Project](https://github.com/jackwish/tflite2onnx/projects/4)

* Now open source with [annocement blog](https://jackwish.net/2020/Convert-TensorFlow-Lite-models-to-ONNX.html).
* [Quantization support](https://github.com/jackwish/tflite2onnx/issues/10) enabled, and tried quantized MobileNetV1 an MobileNetV2.
* Drop [Transpose based layout handling](https://github.com/jackwish/tflite2onnx/issues/2) to save effort of managing quantization.
* More [operators](https://github.com/jackwish/tflite2onnx/blob/master/docs/operator-support-status.md) added, and [tested models](https://github.com/jackwish/tflite2onnx/tree/more-model-test/assets/networks):
  * MobileNetV1
  * MobileNetV2
  * DenseNet
  * EfficientNet
  * MnasNet
  * SqueezeNet
  * NasNet

## v0.2.0

2020-07-15, [Project](https://github.com/jackwish/tflite2onnx/projects/2)

* Operator support of MobileNetV2.
* Infrastructure improvements.
* [Propagation based layout handling](https://github.com/jackwish/tflite2onnx/issues/2).


## v0.1.0

2020-05-24, [Project](https://github.com/jackwish/tflite2onnx/projects/1)

* Model converting Workflow.
* Basic operator support of MobileNetV1.
* [Transpose based layout handling](https://github.com/jackwish/tflite2onnx/issues/2).
