import tflite
from onnx import helper

from .common import T2OBase, logger
from . import tensor
from .op import convert


class Graph(T2OBase):
    def __init__(self, model: tflite.Model, graph: tflite.SubGraph):
        self.ops = []
        self.inputs = []
        self.outputs = []
        self.initializers = []
        logger.debug("Converting...")
        self.tflite = graph
        tensor.registery.clear()

        # operators
        for i in range(graph.OperatorsLength()):
            logger.debug("Converting operator: {}".format(i))
            op_tflite = graph.Operators(i)
            ops = convert(model, graph, op_tflite)
            ops = ops if isinstance(ops, list) else [ops]
            self.ops += ops
            for op in ops:
                self.initializers += op.weights

        # inputs
        logger.debug("Converting inputs...")
        for i in range(graph.InputsLength()):
            index = graph.Inputs(i)
            t = tensor.convert(model, graph, index)
            self.inputs.append(t)

        # outputs
        for i in range(graph.OutputsLength()):
            index = graph.Outputs(i)
            t = tensor.convert(model, graph, index)
            self.outputs.append(t)

        logger.debug("Making ONNX...")
        onodes = [n.onnx for n in self.ops]
        oinputs = [t.onnx for t in self.inputs]
        ooutputs = [t.onnx for t in self.outputs]
        initializer = [t.onnx for t in self.initializers]

        self.onnx = helper.make_graph(onodes, 'pre-alpha', oinputs, ooutputs,
                                      initializer=initializer)
