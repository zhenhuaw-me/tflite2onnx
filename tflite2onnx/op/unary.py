import tflite
from onnx import helper

from ..common import logger
from ..tensor import Tensor
from .op import Operator

OpTypeMapping = {
        tflite.BuiltinOperator.ABS : 'Abs',
        }

class Unary(Operator):
    def __init__(self, model, graph, op):
        logger.debug("[Unary] Converting...")
        self.tflite = op
        opcode = model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
        assert(opcode in OpTypeMapping)
        self.type = OpTypeMapping[opcode]

        assert(op.InputsLength() == 1)
        assert(op.OutputsLength() == 1)

        ti = op.Inputs(0)
        to = Tensor(model, graph, ti)
        self.inputs.append(to)

        ti = op.Outputs(0)
        to = Tensor(model, graph, ti)
        self.outputs.append(to)

        inames = [input.name for input in self.inputs]
        onames = [output.name for output in self.outputs]
        logger.debug("[Unary] Making ONNX...")
        self.onnx = helper.make_node(self.type, inames, onames)
