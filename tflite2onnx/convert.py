import tflite
import onnx
from onnx import helper

from .graph import Graph

def convert(tflite_path: str, onnx_path: str):
    print("tflite: %s", tflite_path)
    print("onnx: %s", onnx_path)
    with open(tflite_path, 'rb') as f:
        buf = f.read()
        im = tflite.Model.GetRootAsModel(buf, 0)

    ig = im.Subgraphs(0)

    ograph = Graph(im, ig)
    omodel = helper.make_model(ograph.onnx, producer_name='tflite2onnx')
    onnx.checker.check_model(omodel)
    onnx.save(omodel, onnx_path)


