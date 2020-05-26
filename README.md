Convert TensorFlow Lite models to ONNX
======================================

![Build and Test](https://github.com/jackwish/tflite2onnx/workflows/Build%20and%20Test/badge.svg)
![Sanity](https://github.com/jackwish/tflite2onnx/workflows/Sanity/badge.svg)

This [`tflite2onnx` package][pypi] converts
TensorFlow Lite (TFLite) models (`*.tflite`) to ONNX models (`*.onnx`).

***This project is under early stage of development, contributions are welcome at [GitHub][github].***


## Usage

### Installation

Simply install via [pip][pypi].
This package requires `tflite`, `onnx` and `numpy` which should be installed automatically.
An executable `tflite2onnx` will be installed also.

```sh
pip install tflite2onnx
```

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

## Resources

* [PyPI page][pypi].
* [GitHub page][github].
* [TensorFlow Lite converter][tf2tflite].


## License

Apache License Version 2.0.

[pypi]: https://pypi.org/project/tflite2onnx/
[github]: https://github.com/jackwish/tflite2onnx
[tf2tflite]: https://www.tensorflow.org/lite/convert
