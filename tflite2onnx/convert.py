import logging
import tflite

from tflite2onnx.model import Model

logger = logging.getLogger('tflite2onnx')


def convert(tflite_path: str, onnx_path: str):
    """Converting TensorFlow Lite models (*tflite) to ONNX models"""

    logger.debug("tflite: %s", tflite_path)
    logger.debug("onnx: %s", onnx_path)
    with open(tflite_path, 'rb') as f:
        buf = f.read()
        im = tflite.Model.GetRootAsModel(buf, 0)

    model = Model(im)
    model.convert()
    model.save(onnx_path)
    logger.info("Converted ONNX model: %s", onnx_path)
