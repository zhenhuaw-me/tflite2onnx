"""Converting TensorFlow Lite models (*.tflite) to ONNX models (*.onnx)"""

from tflite2onnx.common import enableDebugLog
from tflite2onnx.convert import convert

# package metadata
NAME = 'tflite2onnx'
VERSION = '0.3.0'
__version__ = VERSION
DESCRIPTION = "Convert TensorFlow Lite models to ONNX models"

__all__ = [
    convert,
    enableDebugLog,
    NAME,
    VERSION,
    __version__,
    DESCRIPTION,
]
