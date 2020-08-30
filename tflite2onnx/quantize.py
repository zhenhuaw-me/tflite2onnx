import logging
from onnx import TensorProto

from tflite2onnx import tensor
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
    qtensor = tensor.getWithRef(t, qtname, True)
    qtensor.dtype = TensorProto.UINT8
    qtensor.setParsed()

    # create Quantize op
    qop = Quantize(t.model, t.graph, -1)
    qop.name = name_prefix + '_Quantize'
    qop.inputs.append(t)
    qop.outputs.append(qtensor)
    qop.setParsed()
    qop.dequantize()

    # create dequantized tensor
    deqtname = name_prefix + '_dequantized'
    deqtensor = tensor.getWithRef(t, deqtname, True)
    deqtensor.dtype = TensorProto.FLOAT
    deqtensor.setParsed()

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
        master_op.post.insert(0, deqop)
        master_op.post.insert(0, qop)
    elif t.consumers:
        master_op = t.consumers[0]
        master_op.pre.insert(0, deqop)
        master_op.pre.insert(0, qop)
    else:
        assert(False), "No place to add op"

    # link pattern to graph
    for c in t.consumers:
        c.replaceInput(t, deqtensor)
        deqtensor.addConsumer(c)
    t.consumers.clear()
    t.addConsumer(qop)

    return deqtensor
