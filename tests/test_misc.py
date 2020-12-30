import logging
import shrub
from tflite2onnx import getSupportedOperator

shrub.util.formatLogging(logging.DEBUG)


def test_supported_ops():
    assert(len(getSupportedOperator()) > 0)
    assert(getSupportedOperator(0) == 'ADD')


if __name__ == '__main__':
    test_supported_ops()
