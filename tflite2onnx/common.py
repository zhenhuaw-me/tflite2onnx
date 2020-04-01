import logging
from abc import ABC


class BaseABC(ABC):
    """Holding objects of TFLite and ONNX"""
    def __init__(self):
        self.name = 'Unintialized'
        self.tflite = None
        self.onnx = None

    def __str__(self):
        return self.onnx.__str__()


logger = logging.getLogger('tflite2onnx')
