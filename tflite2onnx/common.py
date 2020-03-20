import logging
from abc import ABC

class BaseABC(ABC):
    """Holding objects of TFLite and ONNX"""
    name = "Unintialized"
    tflite = None
    onnx = None

    def __str__(self):
        return self.onnx.__str__()


logger = logging.getLogger('tflite2onnx')
