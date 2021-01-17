"""Converting TensorFlow Lite models (*.tflite) to ONNX models (*.onnx)"""

from tflite2onnx.convert import convert
from tflite2onnx.misc import enableDebugLog, getSupportedOperator

# package metadata
NAME = 'tflite2onnx'
VERSION = '0.3.1'
__version__ = VERSION
DESCRIPTION = 'Convert TensorFlow Lite models to ONNX models. ' \
              'tflite2onnx v%s, https://github.com/jackwish/tflite2onnx' % VERSION

__all__ = [
    convert,
    enableDebugLog,
    getSupportedOperator,
    NAME,
    VERSION,
    __version__,
    DESCRIPTION,
]
