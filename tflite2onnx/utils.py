import tflite
from tflite2onnx.op.common import OpFactory


def enableDebugLog():
    """Dump the logging.DEBUG level log."""
    import logging
    fmt = '%(asctime)s %(levelname).1s [%(name)s][%(filename)s:%(lineno)d] %(message)s'
    logging.basicConfig(format=fmt, level=logging.DEBUG)


def getSupportedOperators():
    """Get the names of the supported TensorFlow Lite operator."""
    opcs = list(OpFactory.registry.keys())
    opcs.sort()
    names = [tflite.BUILTIN_OPCODE2NAME[opc] for opc in opcs]
    return names
