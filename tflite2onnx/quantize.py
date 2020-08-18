import copy
import logging
import tflite
from onnx import helper, TensorProto

from tflite2onnx import tensor
from tflite2onnx.op.operator import Operator
from tflite2onnx.op.quantize import Quantize

logger = logging.getLogger('tflite2onnx')


def handleQuantizationTensor(t):
    """Translate a UINT8 TFLite tensor to Quantize-Dequantize pattern in ONNX.

    As quantization support of ONNX is limited, we currently try to preserve
    the quantization parameters of TFLite model in the resulted ONNX model.
    * All operators are FP operators still.
    * Translate a UINT8 TFLite tensor to Quantize-Dequantize pattern.

    In practice, convert TFLite pattern `[OP A] -> <t1/i> -> [OP B]` to be
    `[OP A] -> <t1/fp32> -> [Quantize] -> <t1'/i> -> [Dequantize] -> <t1''/fp32> -> [OP B]`.
    Where the `[OP A]` or `[OP B]` can be empty if the tensor is an input or
    output w.r.t. the graph. The `<t1>` can be UINT8 mostly, and INT32 for bias.
    To identify bias, the functionality needs to be called by operator?

    We need the `<t1/i>` only because the quantization parameters and
    producer and consumers of it can be easily obtained. For the inserted
    operators, store them in `t1.producers[0].post` or `t1.consumers[0].pre`
    if `<t1/i>` has no producer. Then modify the graph.
    """
    if not t.quantized:
        return t
    logger.debug("Generating quantization pattern for {}".format(t.name))

    t.dequantize()

    if t.is_bias:
        # Bias is INT32, which cannot be processed by Quantize/Dequantize.
        # Fast return here as we need it be float only.
        return t

    name_prefix = 'TFLITE2ONNX_Quant_' + t.name

    # create quantized tensor
    qtname = name_prefix + '_quantized'
    assert(qtname not in tensor.registery)
    qtensor = tensor.Tensor(t.model, t.graph, -1)
    qtensor.name = qtname
    qtensor.dtype = TensorProto.UINT8
    qtensor.layout = copy.deepcopy(t.layout)
    qtensor.shape = copy.deepcopy(t.shape)
    qtensor.scale = copy.deepcopy(t.scale)
    qtensor.zero_point = copy.deepcopy(t.zero_point)
    qtensor.setParsed()
    tensor.registery[qtensor.name] = qtensor

    # create Quantize op
    qop = Quantize(t.model, t.graph, -1)
    qop.name = name_prefix + '_Quantize'
    qop.inputs.append(t)
    qop.outputs.append(qtensor)
    qop.setParsed()
    qop.dequantize()

    # create dequantized tensor
    deqtname = name_prefix + '_dequantized'
    assert(deqtname not in tensor.registery)
    deqtensor = tensor.Tensor(t.model, t.graph, -1)
    deqtensor.name = deqtname
    deqtensor.dtype = TensorProto.FLOAT
    deqtensor.layout = copy.deepcopy(t.layout)
    deqtensor.shape = copy.deepcopy(t.shape)
    deqtensor.scale = copy.deepcopy(t.scale)
    deqtensor.zero_point = copy.deepcopy(t.zero_point)
    deqtensor.setParsed()
    tensor.registery[deqtensor.name] = deqtensor

    # create Dequantize op
    deqop = Quantize(t.model, t.graph, -1)
    deqop.name = name_prefix + '_Dequantize'
    deqop.inputs.append(qtensor)
    deqop.outputs.append(deqtensor)
    deqop.setParsed()
    deqop.dequantize()

    # link local pattern
    qtensor.addProducer(qop)
    qtensor.addConsumer(deqop)
    deqtensor.addProducer(deqop)

    # add Quantize/Dequantize to graph
    if t.producers:
        master_op = t.producers[0]
        master_op.post.append(qop)
        master_op.post.append(deqop)
    elif t.consumers:
        master_op = t.consumers[0]
        master_op.pre.append(qop)
        master_op.pre.append(deqop)
    else:
        assert(False), "No place to add op"

    # link pattern to graph
    for c in t.consumers:
        c.replaceInput(t, deqtensor)
        deqtensor.addConsumer(c)
    t.consumers.clear()
    t.addConsumer(qop)

    return deqtensor
