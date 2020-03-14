import tflite
import onnx
from onnx import helper

from .common import BaseABC
from .graph import Graph

class Model(BaseABC):
    """Everything helps to convert TFLite model to ONNX model"""
    graphs = []

    def __init__(self, model):
        self.tflite = model
        assert(model.SubgraphsLength() == 1)
        for i in range(model.SubgraphsLength()):
            g = model.Subgraphs(i)
            graph = Graph(model, g)
            self.graphs.append(graph)

        self.onnx = helper.make_model(self.graphs[0].onnx, producer_name='tflite2onnx')

    def save(self, path: str):
        onnx.checker.check_model(self.onnx)
        onnx.save(self.onnx, path)
