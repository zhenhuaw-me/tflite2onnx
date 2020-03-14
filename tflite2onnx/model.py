import tflite
from onnx import helper

from .graph import Graph

class Model(object):
    name = None
    graphs = []
    tflite = None
    onnx = None

    def __init__(self, model):
        self.tflite = model
        assert(model.SubgraphsLength() == 1)
        for i in range(model.SubgraphsLength()):
            g = model.Subgraphs(i)
            graph = Graph(model, g)
            self.graphs.append(graph)

        self.onnx = helper.make_model(self.graphs[0].onnx, producer_name='tflite2onnx')
