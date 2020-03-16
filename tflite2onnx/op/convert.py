import tflite

from ..common import logger
from .unary import Unary

OP_CONVERTERS = {
        tflite.BuiltinOperator.ABS : Unary,
        }

def convert(model, graph, op):
    opcode = model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
    if opcode not in OP_CONVERTERS:
        raise NotImplementedError("Unsupported TFLite OP: {}".format(opcode))

    op_converter = OP_CONVERTERS[opcode]

    return op_converter(model, graph, op)

