import logging
from abc import ABC


class T2OBase(ABC):
    """Holding objects of TFLite and ONNX"""
    def __init__(self, model=None, graph=None, index=None):
        self.name = 'Unintialized'
        self.onnx = None    # ONNX object
        self.model = model  # TFLite Model object
        self.graph = graph  # TFLite Graph object
        self.index = index  # Object index of tensor or op
        self.tflite = None  # TFLite object of Operator or Tensor

    def __str__(self):
        return self.onnx.__str__()


logger = logging.getLogger('tflite2onnx')
