import copy
import logging
from onnx import TensorProto

from tflite2onnx.op import Quantize, Dequantize
from tflite2onnx import layout
from tflite2onnx import tensor

logger = logging.getLogger('tflite2onnx')


def createQuantize(ftensor, post_op):
    logger.debug("Creating Quantize node for %s", str(post_op))
    name = 'TFLITE2ONNX_' + ftensor.name + '_' + 'quantized'
    assert(name not in tensor.registery)

    # create tensor
    qtensor = tensor.Tensor(ftensor.model, ftensor.graph, -1)
    qtensor.name = name
    qtensor.dtype = TensorProto.UINT8
    qtensor.layout = copy.deepcopy(ftensor.layout)
    qtensor.shape = copy.deepcopy(ftensor.shape)
    qtensor.scale = copy.deepcopy(ftensor.scale)
    qtensor.zero_point = copy.deepcopy(ftensor.zero_point)
    qtensor.setParsed()
    tensor.registery[qtensor.name] = qtensor

    # create Quantize op
    op = Quantize(post_op.model, post_op.graph, -1)
    op.name = name
    op.inputs.append(ftensor)
    op.outputs.append(qtensor)
    op.setParsed()

    ftensor.addConsumer(op)
    ftensor.removeConsumer(post_op)
    qtensor.addProducer(op)
    qtensor.addConsumer(post_op)
    post_op.replaceInput(ftensor, qtensor)

    return (qtensor, op)


def createDequantize(qtensor, pre_op):
    logger.debug("Creating Dequantize node for %s", str(pre_op))
    name = 'TFLITE2ONNX_' + qtensor.name + '_' + 'dequantized'
    assert(name not in tensor.registery)

    # create tensor
    ftensor = tensor.Tensor(qtensor.model, qtensor.graph, -1)
    ftensor.name = name
    ftensor.dtype = TensorProto.FLOAT
    ftensor.layout = copy.deepcopy(qtensor.layout)
    ftensor.shape = copy.deepcopy(qtensor.shape)
    ftensor.scale = copy.deepcopy(qtensor.scale)
    ftensor.zero_point = copy.deepcopy(qtensor.zero_point)
    ftensor.setParsed()
    tensor.registery[qtensor.name] = ftensor

    # create Dequantize op
    op = Dequantize(pre_op.model, pre_op.graph, -1)
    op.name = name
    op.inputs.append(qtensor)
    op.outputs.append(ftensor)
    op.setParsed()

    for c in qtensor.consumers:
        c.replaceInput(qtensor, ftensor)
        ftensor.addConsumer(c)
    ftensor.addProducer(op)
    qtensor.addConsumer(op)

    return (ftensor, op)
