import tflite

from .unary import Unary
from .softmax import Softmax
from .binary import Binary
from .pooling import AveragePool
from .transpose import Transpose


OP_CONVERTERS = {
        tflite.BuiltinOperator.ABS      : Unary,        # noqa: E203
        tflite.BuiltinOperator.SOFTMAX  : Softmax,      # noqa: E203
        tflite.BuiltinOperator.ADD      : Binary,       # noqa: E203
        tflite.BuiltinOperator.AVERAGE_POOL_2D      : AveragePool,       # noqa: E203
        tflite.BuiltinOperator.TRANSPOSE      : Transpose,       # noqa: E203
}


def convert(model, graph, op):
    opcode = model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
    if opcode not in OP_CONVERTERS:
        raise NotImplementedError("Unsupported TFLite OP: {}".format(opcode))

    op_converter = OP_CONVERTERS[opcode]

    if opcode == tflite.BuiltinOperator.AVERAGE_POOL_2D:
        cvt = op_converter(model, graph, op)
        return cvt.convert(model, graph, op)
    else:
        return op_converter(model, graph, op)
