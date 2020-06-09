Convert TensorFlow Lite models to ONNX
======================================

![Build and Test](https://github.com/jackwish/tflite2onnx/workflows/Build%20and%20Test/badge.svg)
![Sanity](https://github.com/jackwish/tflite2onnx/workflows/Sanity/badge.svg)

This [`tflite2onnx` package][pypi] converts
TensorFlow Lite (TFLite) models (`*.tflite`) to ONNX models (`*.onnx`).

***This project is under early stage of development, [contributions are welcome](#contributing).***


## Usage

### Installation

Simply install via [pip][pypi] `pip install tflite2onnx`.

If you'd like to convert a TensorFlow model (`*.pb`) to ONNX, you may try
[`tf2onnx`](https://github.com/onnx/tensorflow-onnx). Or, you can firstly
[convert][tf2tflite] TensorFlow model (`*.pb`)
to TensorFlow Lite models (`*.tflite`), and then convert to ONNX using this tool.

### Python Interface

```py
import tflite2onnx

tflite_path = '/path/to/original/tflite/model'
onnx_path = '/path/to/save/converted/onnx/model'

tflite2onnx.convert(tflite_path, onnx_path)
```

### Command Line Interface

```sh
tflite2onnx /path/to/original/tflite/model /path/to/save/converted/onnx/model
```

## Contributing

Any contributing are welcome to this tool.

* **Report bugs**: If you think that something is wrong, please using the [bug report](https://github.com/jackwish/tflite2onnx/issues/new?assignees=&labels=bug&template=bug-report.md&title=) issue template.
* **Request operator support**: If you find that some operators are not supported yet, you may comment on [Operator Support Status](https://github.com/jackwish/tflite2onnx/issues/3) with TensorFlow Lite model (which contains that operator only) attached.
* **Enable new operator**: It would be great if you can help to enable new operators, please join us with [How to enable new operator](docs/how-to-enable-new-operator.md).
* **Others**: Please feel free to open discussions if you have any great idea to improve this tool.


## Resources

* [PyPI page][pypi].
* [GitHub page][github].
* [TensorFlow Lite converter][tf2tflite].


## License

Apache License Version 2.0.

[pypi]: https://pypi.org/project/tflite2onnx/
[github]: https://github.com/jackwish/tflite2onnx
[tf2tflite]: https://www.tensorflow.org/lite/convert
