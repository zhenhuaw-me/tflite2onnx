import copy
import logging
from onnx import TensorProto

from tflite2onnx.op import Quantize
from tflite2onnx import layout
from tflite2onnx import tensor

logger = logging.getLogger('tflite2onnx')


def createQuantize(ftensor, post_op):
    logger.debug("Creating Quantize node for ", str(post_op))
    name = 'TFLITE2ONNX_' + ftensor.name + '_' + 'quantized'
    assert(name not in tensor.registery)

    # create tensor
    qtensor = tensor.Tensor(ftensor.model, ftensor.graph, -1)
    qtensor.name = name
    qtensor.dtype = TensorProto.UINT8
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
    post_op.replaceInput(ftensor, qtensor)

    return (qtensor, op)
