import logging
from onnx import TensorProto

from tflite2onnx.op.quantize import Quantize

logger = logging.getLogger('tflite2onnx')


def handleQuantizationTensor(TFactory, t):
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
    qtensor = TFactory.getWithRef(t, qtname, True)
    qtensor.dtype = TensorProto.UINT8
    qtensor.setParsed()

    # create Quantize op
    qop = Quantize(TFactory, -1)
    qop.name = name_prefix + '_Quantize'
    qop.inputs.append(t)
    qop.outputs.append(qtensor)
    qop.setParsed()
    qop.dequantize()

    # create dequantized tensor
    deqtname = name_prefix + '_dequantized'
    deqtensor = TFactory.getWithRef(t, deqtname, True)
    deqtensor.dtype = TensorProto.FLOAT
    deqtensor.setParsed()

    # create Dequantize op
    deqop = Quantize(TFactory, -1)
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


def foldFP16QuantPattern(ops):
    """Fold TFLite FP16 quantization pattern.

    * `FP16 weights - Dequantize - FP32 tensor - Conv` -> `FP32 weights - Conv`
    * `FP16 input - Dequantize - FP32 tensor - Conv` -> `FP32 input - Conv`
    """
    logger.debug("FP16 Quant Folder: Folding FP16 quantization subgraph across graph...")

    # Using `Graph.ops` in this part as these operators should be raw TFLite ones
    fp16deqs = [op for op in ops if op.type == 'DequantizeLinear' and op.inputs[0].dtype is TensorProto.FLOAT16]  # noqa: E501

    count = 0
    for dep in fp16deqs:
        logger.debug("FP16 Quant Folder: Folding FP16-Quant for op %s", dep.name)
        fp16i = dep.inputs[0]
        fp16i.dtype = TensorProto.FLOAT
        if fp16i.isInitializer:
            fp16i.data = fp16i.data.astype('float32')
        fp32i = fp16i

        # attach the casted fp32 tensor to the op that consumes the output of the Dequantize
        fp32o = dep.outputs[0]
        for op in fp32o.consumers:
            op.replaceInput(fp32o, fp32i)
            fp32i.addConsumer(op)
        fp32i.removeConsumer(dep)

        # remove Dequantize operator
        ops.remove(dep)
        # the unneeded tensos will be removed in graph automatically
        # graph.value_info.remove(fp32o)
        count += 1

    if count > 0:
        logger.info("FP16 Quant Folder: %d FP16 Quant Pattern are folded!", count)
