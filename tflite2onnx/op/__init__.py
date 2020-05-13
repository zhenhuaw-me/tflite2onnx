import tflite

from tflite2onnx.op.operator import Operator    # noqa: F401
from tflite2onnx.op.unary import Unary
from tflite2onnx.op.softmax import Softmax
from tflite2onnx.op.binary import Binary
from tflite2onnx.op.pooling import AveragePool
from tflite2onnx.op.transpose import Transpose
from tflite2onnx.op.conv import Conv
from tflite2onnx.op.activation import ReLU
from tflite2onnx.op.reshape import Reshape


OP_CONVERTERS = {
    tflite.BuiltinOperator.ABS: Unary,
    tflite.BuiltinOperator.SOFTMAX: Softmax,
    tflite.BuiltinOperator.ADD: Binary,
    tflite.BuiltinOperator.AVERAGE_POOL_2D: AveragePool,
    tflite.BuiltinOperator.TRANSPOSE: Transpose,
    tflite.BuiltinOperator.CONV_2D: Conv,
    tflite.BuiltinOperator.RELU: ReLU,
    tflite.BuiltinOperator.RELU6: ReLU,
    tflite.BuiltinOperator.RESHAPE: Reshape,
    tflite.BuiltinOperator.DEPTHWISE_CONV_2D: Conv,
}


def getOp(model, graph, index):
    op = graph.Operators(index)
    opcode = model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
    if opcode not in OP_CONVERTERS:
        raise NotImplementedError("Unsupported TFLite OP: {}".format(opcode))

    op_converter = OP_CONVERTERS[opcode]
    return op_converter(model, graph, index)
