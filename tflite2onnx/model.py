import tflite
import onnx
from onnx import helper

from .common import T2OBase, logger
from .graph import Graph


class Model(T2OBase):
    """Everything helps to convert TFLite model to ONNX model"""
    def __init__(self, model: tflite.Model):
        super().__init__(model)
        self.tflite = model
        self.setInited()

    def parse(self):
        graph_count = self.model.SubgraphsLength()
        if (graph_count != 1):
            raise NotImplementedError("ONNX supports one graph per model only, while TFLite has ",
                                      graph_count)
        self.setParsed()

    def convert(self):
        tflg = self.model.Subgraphs(0)
        graph = Graph(self.model, tflg)
        self.onnx = helper.make_model(graph.onnx, producer_name='tflite2onnx')
        self.setConverted()

    def save(self, path: str):
        logger.debug("saving model as %s", path)
        assert(self.onnx is not None)
        onnx.checker.check_model(self.onnx)
        onnx.save(self.onnx, path)
