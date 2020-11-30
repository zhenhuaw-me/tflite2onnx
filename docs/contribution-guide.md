Contribution Guide
==================

Welcome and thank you for reaching this contribution guide.
Materials are split into sections, just jump to topics you are interested in.


## Reporting Issues

* If something seems wrong to you, [report bugs](https://github.com/jackwish/tflite2onnx/issues/new?assignees=&labels=bug&template=bug-report.md&title=).
* If some operators are not supported yet, you may [request a new operator](https://github.com/jackwish/tflite2onnx/issues/new?assignees=&labels=operator%2C+help+wanted&template=request-operator.md&title=Operator+request%3A).
* Feel free to open any other related discussions.

It's high recommended to attach a narrow down-ed TFLite model and
debug logs (generate it with `tflite2onnx.enableDebugLog()`).


## Contributing Code

We work on enabling operators most of the time.
In this way, there is [a dedicated step-by-step guide](docs/how-to-enable-new-operator.md)
to help you enable new operators.

Please help to upstream your operator enabling.
_Unus pro omnibus, omnes pro uno._

We have GitHub Actions based CI for pull requests.
I know sometimes it's annoying but it's very important as it help us to protect the code.
In this way we can keep the quality and save time of debugging.

In general we need:
* Dedicated test of the new operator.
  * It would be great if we can have models to test different attributes.
  * `pytest` at root directory to run all test.
* Clean code.
* Code style.
  * `flake8` at root directory to check.
* No significant code coverage drop (guarded by `CodeCov`).
  * Automatically checked when open/update PR.

Like many other python packages, you can set `PYTHONPATH` to `tflite2onnx`
instead of building and installing to try our your changed.
