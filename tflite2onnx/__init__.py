"""Converting TensorFlow Lite models (*.tflite) to ONNX models (*.onnx)"""

from tflite2onnx.convert import convert
from tflite2onnx.utils import enableDebugLog, getSupportedOperators

# package metadata
__name__ = 'tflite2onnx'
__version__ = '0.4.0'
DESCRIPTION = "Convert TensorFlow Lite models to ONNX"

__all__ = [
    convert,
    enableDebugLog,
    getSupportedOperators,
    __name__,
    __version__,
    DESCRIPTION,
]
