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
        onodes = []
        oinputs = []
        ooutputs = []

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

            # FIXME: not all op IO are graph IO
            self.outputs += op.outputs

            onodes.append(op.onnx)
            for t in op.outputs:
                ooutputs.append(t.onnx)


        logger.debug("[Graph] Making ONNX...")
        oinputs = [t.onnx for t in self.inputs]
        self.onnx = helper.make_graph(onodes, 'pre-alpha', oinputs, ooutputs)



