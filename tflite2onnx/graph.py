import tflite
from onnx import helper

from .op.mapping import Operator

class Graph(object):
    name = None
    ops = []
    inputs = []
    outputs = []
    onnx = None

    def __init__(self, model, graph):
        onodes = []
        oinputs = []
        ooutputs = []
        for i in range(graph.OperatorsLength()):
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

            self.onnx = helper.make_graph(onodes, 'pre-alpha', oinputs, ooutputs)



