import argparse
import logging
import os

import tflite
import tflite2onnx
from tflite2onnx.model import Model
from tflite2onnx.common import LayoutApproach

logger = logging.getLogger('tflite2onnx')


def convert(tflite_path: str, onnx_path: str,
            layout_approach=LayoutApproach.DEFAULT):
    """Converting TensorFlow Lite models (*tflite) to ONNX models"""

    if not os.path.exists(tflite_path):
        raise ValueError("Invalid TFLite model path (%s)!" % tflite_path)

    logger.debug("tflite: %s", tflite_path)
    logger.debug("onnx: %s", onnx_path)
    with open(tflite_path, 'rb') as f:
        buf = f.read()
        im = tflite.Model.GetRootAsModel(buf, 0)

    model = Model(im)
    model.convert(layout_approach)
    model.save(onnx_path)
    logger.info("Converted ONNX model: %s", onnx_path)


def cmd_convert():
    parser = argparse.ArgumentParser(description=tflite2onnx.DESCRIPTION)
    parser.add_argument('tflite_path', help="Path to the input TFLite mode")
    parser.add_argument('onnx_path', help="Path to save the converted ONNX mode")

    args = parser.parse_args()
    convert(args.tflite_path, args.onnx_path)
