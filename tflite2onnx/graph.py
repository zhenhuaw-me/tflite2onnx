import tflite
from onnx import helper

from .common import T2OBase, logger
from . import tensor
from .op import getOp


class Graph(T2OBase):
    def __init__(self, model: tflite.Model, graph: tflite.SubGraph):
        super().__init__(model, graph)
        self.ops = []
        self.inputs = []
        self.outputs = []
        self.initializer = dict()
        self.value_info = dict()
        self.tflite = graph
        tensor.registery.clear()
        self.setInited()

    def parse(self):
        logger.debug("Parsing the Graph...")
        # operators
        for i in range(self.graph.OperatorsLength()):
            logger.debug("Parsing operator: {}".format(i))
            op = getOp(self.model, self.graph, i)
            op.parse()
            self.ops.append(op)

        # inputs
        logger.debug("Parsing inputs...")
        for i in range(self.graph.InputsLength()):
            # FIXME: assert they have been created.
            index = self.graph.Inputs(i)
            t = tensor.get(self.model, self.graph, index)
            self.inputs.append(t)

        # outputs
        for i in range(self.graph.OutputsLength()):
            index = self.graph.Outputs(i)
            t = tensor.get(self.model, self.graph, index)
            self.outputs.append(t)

        for op in self.ops:
            pass


        self.setParsed()

    def propagate(self):
        logger.debug("Propagating...")



        for op in self.ops:
            logger.debug("[OP] %s", str(op))
        for t in self.inputs:
            logger.debug("[Inputs] %s", str(t))
        for t in self.initializer:
            logger.debug("[Initializer] %s", str(t))
        for t in self.outputs:
            logger.debug("[Outputs] %s", str(t))
        self.setPropagated()

    def convert(self):
        logger.debug("Converting...")

        for op in self.ops:
            op.convert()

        logger.debug("Making ONNX...")
        onodes = [n.onnx for n in self.ops]
        oinputs = [t.onnx for t in self.inputs]
        ooutputs = [t.onnx for t in self.outputs]
        initializer = [t.onnx for n,t in self.initializer.items()]
        value_info = [t.onnx for n,t in self.value_info.items()]

        self.onnx = helper.make_graph(onodes, 'pre-alpha', oinputs, ooutputs,
                                      initializer=initializer, value_info=value_info)
        self.setConverted()
