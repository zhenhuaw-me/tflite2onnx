import copy
import logging
import tflite
from onnx import helper, TensorProto

from tflite2onnx import tensor
from tflite2onnx.op.operator import Operator

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
        return
    logger.debug("Generating quantization pattern for {}".format(t.name))
