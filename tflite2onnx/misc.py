from tflite2onnx.op.common import OpFactory
from tflite2onnx.tflops import TFLITE_OP_NAME_LIST


def enableDebugLog():
    """Dump the logging.DEBUG level log."""
    import logging
    fmt = '%(asctime)s %(levelname).1s [%(name)s][%(filename)s:%(lineno)d] %(message)s'
    logging.basicConfig(format=fmt, level=logging.DEBUG)


def getSupportedOperator(opc=None):
    """Get the name(s) of the supported TensorFlow Lite operator.

    Args:
        opc(int): The op code of the TensorFlow Lite operator.
    Return the full list (names) of supported TensorFlow Lite operator if
    opc is not specified, otherwise the operator name of the op code, e.g. CONV.
    """
    if opc is None:
        opcs = list(OpFactory.registry.keys())
        opcs.sort()
        names = [TFLITE_OP_NAME_LIST[opc] for opc in opcs]
        return names
    else:
        if (not isinstance(opc, int)) or (opc >= len(TFLITE_OP_NAME_LIST)):
            raise ValueError("Op code %d is unknown!" % opc)
        return TFLITE_OP_NAME_LIST[opc]
