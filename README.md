tflite2onnx - Convert TensorFlow Lite models to ONNX
====================================================

[![Build and Test](https://github.com/jackwish/tflite2onnx/workflows/Build%20and%20Test/badge.svg)](https://github.com/jackwish/tflite2onnx/actions?query=workflow%3A%22Build+and+Test%22)
[![Sanity](https://github.com/jackwish/tflite2onnx/workflows/Sanity/badge.svg)](https://github.com/jackwish/tflite2onnx/actions?query=workflow%3ASanity)
[![Coverage](https://codecov.io/gh/jackwish/tflite2onnx/branch/master/graph/badge.svg)](https://codecov.io/gh/jackwish/tflite2onnx)

`tflite2onnx` converts TensorFlow Lite (TFLite) models (`*.tflite`) to ONNX models (`*.onnx`),
with data layout and quantization semantic properly handled (check the [introduction blog][intro] for detail).

**Highlights**


* If you'd like to convert a TensorFlow model (frozen graph `*.pb`, `SavedModel`
or whatever) to ONNX, try [`tf2onnx`](https://github.com/onnx/tensorflow-onnx).
Or, you can firstly [convert][tf2tflite] it to a TFLite (`*.tflite`) model,
and then convert the TFLite model to ONNX.

* Microsoft has implemented another _TensorFlow Lite to ONNX model converter_ in `tf2onnx`
[at Feb 2021](https://github.com/onnx/sigs/blob/master/converters/meetings/019-20210212.md)
(we open sourced `tflite2onnx` in May 2020). `tf2onnx` seems to able to convert Quantization
just like us, and it seems able to convert RNN networks which we are not supported yet.
Please try `tf2onnx --tflite` if `tflite2onnx` missing any functionality.


## Installation

Install via [pip][pypi] `pip install tflite2onnx`.

Or install from source to get latest features (please try out with [virtualenv](https://virtualenv.pypa.io)):

1. Download the repo: `git clone https://github.com/jackwish/tflite2onnx.git && cd tflite2onnx`
2. Build the package: `./scripts/build-wheel.sh`
3. Install the built package: `pip install assets/dist/tflite2onnx-*.whl`

Or you can just add the code tree to your `$PYTHONPATH`.
(Command line tool is not avaiable in this mode.)

```sh
export PYTHONPATH=$(pwd):${PYTHONPATH}
```


## Usage

### Python Interface

```py
import tflite2onnx

tflite_path = '/path/to/original/tflite/model'
onnx_path = '/path/to/save/converted/onnx/model'

tflite2onnx.convert(tflite_path, onnx_path)
```

`tflite2onnx` now supports *explicit layout*, check the
[test example](https://github.com/jackwish/tflite2onnx/blob/master/tests/test_explicit_layout.py).


### Command Line

```sh
tflite2onnx /path/to/original/tflite/model /path/to/save/converted/onnx/model
```


## Documentation

* [FAQ](docs/faq.md)
* [Release note](docs/release-notes.md)
* [Contribution guide](docs/contribution-guide.md)
* [Introduction blog - the background, design and implementation][intro]
* [How to enable a new operator](docs/how-to-enable-new-operator.md)
* [Data layout semantic](docs/data-layout-semantic.md)


## Contributing

* If something seems wrong to you, [report bugs](https://github.com/jackwish/tflite2onnx/issues/new?assignees=&labels=bug&template=bug-report.md&title=).
* If some operators are not supported yet, you may [request a new operator](https://github.com/jackwish/tflite2onnx/issues/new?assignees=&labels=operator%2C+help+wanted&template=request-operator.md&title=Operator+request%3A).
* It would be great if you can help to enable new operators, please join us with [How to enable a new operator](docs/how-to-enable-new-operator.md).
* Feel free to open any other related discussions.

Check [contribution guide](docs/contribution-guide.md) for more.


## License

Apache License Version 2.0.

[intro]: https://jackwish.net/2020/Convert-TensorFlow-Lite-models-to-ONNX.html
[pypi]: https://pypi.org/project/tflite2onnx
[github]: https://github.com/jackwish/tflite2onnx
[tf2tflite]: https://www.tensorflow.org/lite/convert
