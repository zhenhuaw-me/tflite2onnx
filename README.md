tflite2onnx - Convert TensorFlow Lite models to ONNX
====================================================

[![Build and Test](https://github.com/jackwish/tflite2onnx/workflows/Build%20and%20Test/badge.svg)](https://github.com/jackwish/tflite2onnx/actions?query=workflow%3A%22Build+and+Test%22)
[![Sanity](https://github.com/jackwish/tflite2onnx/workflows/Sanity/badge.svg)](https://github.com/jackwish/tflite2onnx/actions?query=workflow%3ASanity)
[![Coverage](https://codecov.io/gh/jackwish/tflite2onnx/branch/master/graph/badge.svg)](https://codecov.io/gh/jackwish/tflite2onnx)

`tflite2onnx` converts TensorFlow Lite (TFLite) models (`*.tflite`) to ONNX models (`*.onnx`),
with data layout and quantization semantic properly handled (check the [introduction blog][intro] for detail).

> If you'd like to convert a TensorFlow model (frozen graph `*.pb`, `SavedModel`
or whatever) to ONNX, try [`tf2onnx`](https://github.com/onnx/tensorflow-onnx).
Or, you can firstly [convert][tf2tflite] it to a TFLite (`*.tflite`) model,
and then convert the TFLite model to ONNX.


## Usage

Install via [pip][pypi] `pip install tflite2onnx`.
After installation, you may either try either.

**Python interface**

```py
import tflite2onnx

tflite_path = '/path/to/original/tflite/model'
onnx_path = '/path/to/save/converted/onnx/model'

tflite2onnx.convert(tflite_path, onnx_path)
```

`tflite2onnx` now supports *explicit layout*, check the
[test example](https://github.com/jackwish/tflite2onnx/blob/master/tests/test_explicit_layout.py).

**Command line**

```sh
tflite2onnx /path/to/original/tflite/model /path/to/save/converted/onnx/model
```

## Documents

* [Introduction blog - the background, design and implementation][intro]
* [Release note](docs/release-notes.md)
* [Supported operators](docs/operator-support-status.md) ([Onging status issue](https://github.com/jackwish/tflite2onnx/issues/11))
* [How to enable a new operator](docs/how-to-enable-new-operator.md)


## Contributing

* If you think something is wrong, [report bugs](https://github.com/jackwish/tflite2onnx/issues/new?assignees=&labels=bug&template=bug-report.md&title=).
* If some operators are not supported yet, you may [request a new operator](https://github.com/jackwish/tflite2onnx/issues/new?assignees=&labels=operator%2C+help+wanted&template=request-operator.md&title=Operator+request%3A).
* It would be great if you can help to enable new operators, please join us with [How to enable a new operator](docs/how-to-enable-new-operator.md).
* Feel free to open discussions if you have any great idea to improve this tool.


## License

Apache License Version 2.0.

[intro]: https://jackwish.net/2020/Convert-TensorFlow-Lite-models-to-ONNX.html
[pypi]: https://pypi.org/project/tflite2onnx
[github]: https://github.com/jackwish/tflite2onnx
[tf2tflite]: https://www.tensorflow.org/lite/convert
