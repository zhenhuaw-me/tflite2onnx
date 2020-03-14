import tflite
import onnx
from onnx import helper

from .model import Model


def convert(tflite_path: str, onnx_path: str):
    """Converting TensorFlow Lite models (*tflite) to ONNX models"""

    print("tflite: ", tflite_path)
    print("onnx: ", onnx_path)
    with open(tflite_path, 'rb') as f:
        buf = f.read()
        im = tflite.Model.GetRootAsModel(buf, 0)

    model = Model(im)
    model.save(onnx_path)


