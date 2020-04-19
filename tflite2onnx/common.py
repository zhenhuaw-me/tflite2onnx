import logging
from abc import ABC


class T2OBase(ABC):
    """Holding objects of TFLite and ONNX"""
    def __init__(self):
        self.name = 'Unintialized'
        self.tflite = None
        self.onnx = None

    def __str__(self):
        return self.onnx.__str__()


class TFLiteObjectBase(ABC):
    def __init__(self, model, graph, index):
            self.model = model
            self.graph = graph
            self.index = index  # index of tensor or op


logger = logging.getLogger('tflite2onnx')
