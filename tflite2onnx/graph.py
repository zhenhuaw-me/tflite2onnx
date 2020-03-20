import tflite
from onnx import helper

from .common import BaseABC, logger
from .tensor import create_tensor
from .op import convert

class Graph(BaseABC):
    ops = []
    inputs = []
    outputs = []

    def __init__(self, model: tflite.Model, graph: tflite.SubGraph):
        logger.debug("[Graph] Converting...")
        self.tflite = graph

        # inputs
        logger.debug("[Graph] Converting inputs...")
        for i in range(graph.InputsLength()):
            index = graph.Inputs(i)
            t = create_tensor(model, graph, index)
            self.inputs.append(t)


        # operators
        for i in range(graph.OperatorsLength()):
            logger.debug("[Graph] Converting operator: {}".format(i))
            op_tflite = graph.Operators(i)
            op = convert(model, graph, op_tflite)
            self.ops.append(op)

        # outputs
        for i in range(graph.OutputsLength()):
            index = graph.Outputs(i)
            t = create_tensor(model, graph, index)
            self.outputs.append(t)

        logger.debug("[Graph] Making ONNX...")
        onodes = [n.onnx for n in self.ops]
        oinputs = [t.onnx for t in self.inputs]
        ooutputs = [t.onnx for t in self.outputs]
        self.onnx = helper.make_graph(onodes, 'pre-alpha', oinputs, ooutputs)



