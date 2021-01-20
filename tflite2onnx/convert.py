import argparse
import logging
import os

import tflite
import tflite2onnx
from tflite2onnx.model import Model

logger = logging.getLogger('tflite2onnx')


def convert(tflite_path: str, onnx_path: str, explicit_layouts=None):
    """Converting TensorFlow Lite model (*.tflite) to ONNX model.

    Args:
        tflite_path (str): the path to TFLite model.
        onnx_path (str): the path where to save the converted ONNX model.
        explicit_layouts (dict, optinal): Dict of `str -> tuple(str, str)`.
            For each items, its *tensor name* `->` *tflite layout* and *onnx layout*.
            This can be safely ignored usually - tflite2onnx can handle most
            layout semantic divergence automatically.
    """

    if not os.path.exists(tflite_path):
        raise ValueError("Invalid TFLite model path (%s)!" % tflite_path)
    if os.path.exists(onnx_path):
        logger.warning("ONNX model path (%s) existed!", onnx_path)

    if explicit_layouts:
        for k, v in explicit_layouts.items():
            if not (isinstance(k, str) and isinstance(v, tuple) and
                    (len(v) == 2) and isinstance(v[0], str) or isinstance(v[1], str)):
                raise ValueError("Invalid explicit layouts!")
    else:
        explicit_layouts = dict()

    logger.debug("tflite: %s", tflite_path)
    logger.debug("onnx: %s", onnx_path)
    with open(tflite_path, 'rb') as f:
        buf = f.read()
        im = tflite.Model.GetRootAsModel(buf, 0)

    model = Model(im)
    model.convert(explicit_layouts)
    model.save(onnx_path)
    logger.info("Converted ONNX model: %s", onnx_path)


def cmd_convert():
    description = "tflite2onnx " + tflite2onnx.__version__ + ", " + tflite2onnx.DESCRIPTION
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('tflite_path', help="Path to the input TFLite mode")
    parser.add_argument('onnx_path', help="Path to save the converted ONNX mode")

    args = parser.parse_args()

    convert(args.tflite_path, args.onnx_path)
