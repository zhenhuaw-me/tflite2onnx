import tflite
import onnx
from onnx import helper

from .common import T2OBase, logger
from .graph import Graph


class Model(T2OBase):
    """Everything helps to convert TFLite model to ONNX model"""
    def __init__(self, model: tflite.Model):
        self.graphs = []
        self.tflite = model
        if (model.SubgraphsLength() != 1):
            raise NotImplementedError("ONNX supports one graph per model only, while TFLite has ",
                                      model.SubgraphsLength())

        for i in range(model.SubgraphsLength()):
            g = model.Subgraphs(i)
            graph = Graph(model, g)
            self.graphs.append(graph)

        assert(len(self.graphs) == model.SubgraphsLength())
        self.onnx = helper.make_model(self.graphs[0].onnx, producer_name='tflite2onnx')

    def save(self, path: str):
        logger.debug("save model as %s", path)
        assert(self.onnx is not None)
        onnx.checker.check_model(self.onnx)
        onnx.save(self.onnx, path)
