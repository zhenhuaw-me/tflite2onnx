import tflite
import onnx
from onnx import helper, AttributeProto, GraphProto
from typing import List

from .op.mapping import Operator

def convert(tflite_path: str, onnx_path: str):
    print("tflite: %s", tflite_path)
    print("onnx: %s", onnx_path)
    with open(tflite_path, 'rb') as f:
        buf = f.read()
        im = tflite.Model.GetRootAsModel(buf, 0)

    ig = im.Subgraphs(0)

    ops = []
    # for i in range(ig.OperatorsLength()):
    for i in range(1):
        op = ig.Operators(i)
        n = Operator(im, ig, op)
        ops.append(n)

    opis = []
    opos = []
    onodes = []
    for op in ops:
        for t in op.inputs:
            opis.append(t.onnx)
        for t in op.outputs:
            opos.append(t.onnx)
        onodes.append(op.onnx)

    ograph = helper.make_graph(onodes, 'init model', opis, opos)
    omodel = helper.make_model(ograph, producer_name='tflite2onnx')
    onnx.checker.check_model(omodel)
    onnx.save(omodel, onnx_path)


