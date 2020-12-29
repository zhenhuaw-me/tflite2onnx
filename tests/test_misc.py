import logging
import shrub
from tflite2onnx import getSupportedOperators

shrub.util.formatLogging(logging.DEBUG)


def test_supported_ops():
    assert len(getSupportedOperators()) > 0


if __name__ == '__main__':
    test_supported_ops()
