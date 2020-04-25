import tflite
from onnx import helper

from .common import T2OBase, logger
from . import tensor
from .op import convert


class Graph(T2OBase):
    def __init__(self, model: tflite.Model, graph: tflite.SubGraph):
        super().__init__(model, graph)
        self.ops = []
        self.inputs = []
        self.outputs = []
        self.initializers = []
        self.tflite = graph
        tensor.registery.clear()
        self.setInited()

    def parse(self):
        self.setParsed()

    def buildGraph(self):
        self.setGraphBuilt()

    def propagate(self):
        self.setPropagated()

    def convert(self):
        logger.debug("Converting...")

        # operators
        for i in range(self.graph.OperatorsLength()):
            logger.debug("Converting operator: {}".format(i))
            ops = convert(self.model, self.graph, i)
            ops = ops if isinstance(ops, list) else [ops]
            self.ops += ops
            for op in ops:
                self.initializers += op.weights

        # inputs
        logger.debug("Converting inputs...")
        for i in range(self.graph.InputsLength()):
            index = self.graph.Inputs(i)
            t = tensor.convert(self.model, self.graph, index)
            self.inputs.append(t)

        # outputs
        for i in range(self.graph.OutputsLength()):
            index = self.graph.Outputs(i)
            t = tensor.convert(self.model, self.graph, index)
            self.outputs.append(t)

        logger.debug("Making ONNX...")
        onodes = [n.onnx for n in self.ops]
        oinputs = [t.onnx for t in self.inputs]
        ooutputs = [t.onnx for t in self.outputs]
        initializer = [t.onnx for t in self.initializers]

        self.onnx = helper.make_graph(onodes, 'pre-alpha', oinputs, ooutputs,
                                      initializer=initializer)
        self.setConverted()
