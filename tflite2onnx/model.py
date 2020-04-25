import tflite
import onnx
from onnx import helper

from .common import T2OBase, logger, Status
from .graph import Graph


class Model(T2OBase):
    """Everything helps to convert TFLite model to ONNX model"""
    def __init__(self, model: tflite.Model):
        super().__init__(model)
        self.tflite = model
        self.graphes = []
        self.setInited()

    def parse(self):
        graph_count = self.model.SubgraphsLength()
        if (graph_count != 1):
            raise NotImplementedError("ONNX supports one graph per model only, while TFLite has ",
                                      graph_count)
        tflg = self.model.Subgraphs(0)
        graph = Graph(self.model, tflg)
        self.graphes.append(graph)

        for g in self.graphes:
            g.parse()

        self.setParsed()

    def buildGraph(self):
        for g in self.graphes:
            g.buildGraph()
        self.setGraphBuilt()

    def propagate(self):
        for g in self.graphes:
            g.propagate()
        self.setPropagated()

    def convert(self):
        self.parse()
        self.buildGraph()
        self.propagate()
        for g in self.graphes:
            g.convert()
        self.onnx = helper.make_model(self.graphes[0].onnx, producer_name='tflite2onnx')
        self.setConverted()

    def save(self, path: str):
        logger.debug("saving model as %s", path)
        assert(self.status is Status.CONVERTED)
        assert(self.onnx is not None)
        onnx.checker.check_model(self.onnx)
        onnx.save(self.onnx, path)
