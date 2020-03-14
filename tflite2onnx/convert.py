import tflite
import onnx
from onnx import helper

from .model import Model


def convert(tflite_path: str, onnx_path: str):
    print("tflite: %s", tflite_path)
    print("onnx: %s", onnx_path)
    with open(tflite_path, 'rb') as f:
        buf = f.read()
        im = tflite.Model.GetRootAsModel(buf, 0)

    model = Model(im)
    onnx.checker.check_model(model.onnx)
    onnx.save(model.onnx, onnx_path)


