import tflite
from onnx import helper

from .common import BaseABC, logger
from .op.mapping import Operator

class Graph(BaseABC):
    ops = []
    inputs = []
    outputs = []

    def __init__(self, model: tflite.Model, graph: tflite.SubGraph):
        logger.debug("[Graph] Converting...")
        self.tflite = graph
        onodes = []
        oinputs = []
        ooutputs = []

        for i in range(graph.OperatorsLength()):
            logger.debug("[Graph] Converting operator: {}".format(i))
            op_tflite = graph.Operators(i)
            op = Operator(model, graph, op_tflite)
            self.ops.append(op)

            # FIXME: not all op IO are graph IO
            self.inputs += op.inputs
            self.outputs += op.outputs

            onodes.append(op.onnx)
            for t in op.inputs:
                oinputs.append(t.onnx)
            for t in op.outputs:
                ooutputs.append(t.onnx)

        logger.debug("[Graph] Making ONNX...")
        self.onnx = helper.make_graph(onodes, 'pre-alpha', oinputs, ooutputs)



