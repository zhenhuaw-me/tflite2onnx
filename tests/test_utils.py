import logging
import shrub
from tflite2onnx import getSupportedOperators

shrub.util.formatLogging(logging.DEBUG)


def test_supported_ops():
    ops = getSupportedOperators()
    assert(len(ops) > 0)
    assert(ops[0] == 'ADD')


if __name__ == '__main__':
    test_supported_ops()
