import tflite

from tflite2onnx.op.operator import Operator
from tflite2onnx.op.unary import Unary
from tflite2onnx.op.softmax import Softmax
from tflite2onnx.op.binary import Binary
from tflite2onnx.op.pooling import AveragePool
from tflite2onnx.op.transpose import Transpose
from tflite2onnx.op.conv import Conv2D
from tflite2onnx.op.activation import ReLU
from tflite2onnx.op.reshape import Reshape


OP_CONVERTERS = {
        tflite.BuiltinOperator.ABS              : Unary,        # noqa: E203
        tflite.BuiltinOperator.SOFTMAX          : Softmax,      # noqa: E203
        tflite.BuiltinOperator.ADD              : Binary,       # noqa: E203
        tflite.BuiltinOperator.AVERAGE_POOL_2D  : AveragePool,  # noqa: E203
        tflite.BuiltinOperator.TRANSPOSE        : Transpose,    # noqa: E203
        tflite.BuiltinOperator.CONV_2D          : Conv2D,       # noqa: E203
        tflite.BuiltinOperator.RELU             : ReLU,         # noqa: E203
        tflite.BuiltinOperator.RELU6            : ReLU,         # noqa: E203
        tflite.BuiltinOperator.RESHAPE          : Reshape,         # noqa: E203
}


def getOp(model, graph, index):
    op = graph.Operators(index)
    opcode = model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
    if opcode not in OP_CONVERTERS:
        raise NotImplementedError("Unsupported TFLite OP: {}".format(opcode))

    op_converter = OP_CONVERTERS[opcode]
    return op_converter(model, graph, index)
