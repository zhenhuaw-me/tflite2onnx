import tflite
from onnx import helper

from ..common import BaseABC, logger
from ..tensor import Tensor

TFLITE2ONNX = {
        tflite.BuiltinOperator.ABS : 'Abs',
        }

class Operator(BaseABC):
    type = None
    inputs = []
    outputs = []

    def __init__(self, model, graph, op):
        logger.debug("[Operator] Converting...")
        self.tflite = op
        opcode = model.OperatorCodes(op.OpcodeIndex())
        assert(opcode.BuiltinCode() in TFLITE2ONNX)
        self.type = TFLITE2ONNX[opcode.BuiltinCode()]

        for i in range(op.InputsLength()):
            ti = op.Inputs(i)
            to = Tensor(model, graph, ti)
            self.inputs.append(to)

        for i in range(op.OutputsLength()):
            ti = op.Outputs(i)
            to = Tensor(model, graph, ti)
            self.outputs.append(to)

        inputs_name = [input.name for input in self.inputs]
        outputs_name = [output.name for output in self.outputs]
        logger.debug("[Operator] Making ONNX...")
        self.onnx = helper.make_node(self.type, inputs_name, outputs_name)
