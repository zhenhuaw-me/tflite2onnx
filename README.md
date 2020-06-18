Converting TensorFlow Lite models to ONNX
=========================================

![Build and Test](https://github.com/jackwish/tflite2onnx/workflows/Build%20and%20Test/badge.svg)
![Sanity](https://github.com/jackwish/tflite2onnx/workflows/Sanity/badge.svg)

`tflite2onnx` converts TensorFlow Lite (TFLite) models (`*.tflite`) to ONNX models (`*.onnx`).

***This project is under early stage of development, [contributions are welcome](#contributing).***


## Usage

> If you'd like to convert a TensorFlow model (`*.pb`) to ONNX, you may try
[`tf2onnx`](https://github.com/onnx/tensorflow-onnx). Or, you can firstly
[convert][tf2tflite] TensorFlow model (`*.pb`)
to TensorFlow Lite models (`*.tflite`), and then convert to ONNX using this tool.

Install via [pip][pypi] `pip install tflite2onnx`.
After installation, you may either try either.

**Python interface**
```py
import tflite2onnx

tflite_path = '/path/to/original/tflite/model'
onnx_path = '/path/to/save/converted/onnx/model'

tflite2onnx.convert(tflite_path, onnx_path)
```

**Command line**
```sh
tflite2onnx /path/to/original/tflite/model /path/to/save/converted/onnx/model
```


## Contributing

Any contribution is welcome to this tool.

* If you think something is wrong, [report bugs](https://github.com/jackwish/tflite2onnx/issues/new?assignees=&labels=bug&template=bug-report.md&title=).
* If you find that some operators are not supported yet, you may [request new operator](https://github.com/jackwish/tflite2onnx/issues/new?assignees=&labels=operator%2C+help+wanted&template=request-operator.md&title=Operator+request%3A).
* It would be great if you can help to enable new operators, please join us with [How to enable new operator](docs/how-to-enable-new-operator.md).
* Feel free to open discussions if you have any great idea to improve this tool.


## Resources

* [PyPI page][pypi].
* [GitHub page][github].
* [TensorFlow Lite converter][tf2tflite].


## License

Apache License Version 2.0.

[pypi]: https://pypi.org/project/tflite2onnx/
[github]: https://github.com/jackwish/tflite2onnx
[tf2tflite]: https://www.tensorflow.org/lite/convert
