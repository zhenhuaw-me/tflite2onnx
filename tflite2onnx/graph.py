import tflite
from onnx import helper

from .common import BaseABC, logger
from .tensor import create_tensor, TensorMapping
from .op import convert


class Graph(BaseABC):
    ops = []
    inputs = []
    outputs = []

    def __init__(self, model: tflite.Model, graph: tflite.SubGraph):
        logger.debug("Converting...")
        self.tflite = graph
        TensorMapping.clear()

        # inputs
        logger.debug("Converting inputs...")
        for i in range(graph.InputsLength()):
            index = graph.Inputs(i)
            t = create_tensor(model, graph, index)
            self.inputs.append(t)

        # operators
        for i in range(graph.OperatorsLength()):
            logger.debug("Converting operator: {}".format(i))
            op_tflite = graph.Operators(i)
            op = convert(model, graph, op_tflite)
            self.ops.append(op)

        # outputs
        for i in range(graph.OutputsLength()):
            index = graph.Outputs(i)
            t = create_tensor(model, graph, index)
            self.outputs.append(t)

        logger.debug("Making ONNX...")
        onodes = [n.onnx for n in self.ops]
        oinputs = [t.onnx for t in self.inputs]
        ooutputs = [t.onnx for t in self.outputs]
        self.onnx = helper.make_graph(onodes, 'pre-alpha', oinputs, ooutputs)
