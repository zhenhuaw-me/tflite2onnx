import tflite

from tflite2onnx.op.activation import ReLU
from tflite2onnx.op.binary import Binary
from tflite2onnx.op.concat import Concat
from tflite2onnx.op.conv import Conv
from tflite2onnx.op.fullyconnected import FullyConnected
from tflite2onnx.op.common import Operator    # noqa: F401
from tflite2onnx.op.padding import Padding
from tflite2onnx.op.pooling import Pooling
from tflite2onnx.op.quantize import Quantize
from tflite2onnx.op.reduce import Reduce
from tflite2onnx.op.reshape import Reshape
from tflite2onnx.op.slice import Slice
from tflite2onnx.op.softmax import Softmax
from tflite2onnx.op.split import Split
from tflite2onnx.op.transpose import Transpose
from tflite2onnx.op.unary import Unary


OP_CONVERTERS = {
    tflite.BuiltinOperator.ABS: Unary,
    tflite.BuiltinOperator.ADD: Binary,
    tflite.BuiltinOperator.AVERAGE_POOL_2D: Pooling,
    tflite.BuiltinOperator.CONCATENATION: Concat,
    tflite.BuiltinOperator.CONV_2D: Conv,
    tflite.BuiltinOperator.DEPTHWISE_CONV_2D: Conv,
    tflite.BuiltinOperator.DEQUANTIZE: Quantize,
    tflite.BuiltinOperator.FULLY_CONNECTED: FullyConnected,
    tflite.BuiltinOperator.MAX_POOL_2D: Pooling,
    tflite.BuiltinOperator.MEAN: Reduce,
    tflite.BuiltinOperator.MUL: Binary,
    tflite.BuiltinOperator.PAD: Padding,
    tflite.BuiltinOperator.QUANTIZE: Quantize,
    tflite.BuiltinOperator.RELU6: ReLU,
    tflite.BuiltinOperator.RELU: ReLU,
    tflite.BuiltinOperator.RESHAPE: Reshape,
    tflite.BuiltinOperator.SOFTMAX: Softmax,
    tflite.BuiltinOperator.SPLIT: Split,
    tflite.BuiltinOperator.STRIDED_SLICE: Slice,
    tflite.BuiltinOperator.TRANSPOSE: Transpose,
}


def getOp(model, graph, tregistry, index):
    op = graph.Operators(index)
    opcode = model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
    if opcode not in OP_CONVERTERS:
        raise NotImplementedError("Unsupported TFLite OP: {}".format(opcode))

    op_converter = OP_CONVERTERS[opcode]
    return op_converter(model, graph, tregistry, index)
