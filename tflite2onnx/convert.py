import os
import sys
import tflite
import onnx
from onnx import helper, AttributeProto, GraphProto
from typing import List

from .tensor import Tensor


OPTYPE_TFLITE2ONNX = {
        tflite.BuiltinOperator.ABS : 'Abs',
        }

def convert(tflite_path: str, onnx_path: str):
    print("tflite: %s", tflite_path)
    print("onnx: %s", onnx_path)
    with open(tflite_path, 'rb') as f:
        buf = f.read()
        im = tflite.Model.GetRootAsModel(buf, 0)

    ig = im.Subgraphs(0)

    onodes = []
    # for i in range(ig.OperatorsLength()):
    for i in range(1):
        op = ig.Operators(i)
        code = im.OperatorCodes(op.OpcodeIndex())
        assert(code.BuiltinCode() == tflite.BuiltinOperator.ABS)

        # inputs/output
        opis = []
        opins = []
        for ii in range(op.InputsLength()):
            ti = op.Inputs(ii)
            to = Tensor(im, ig, ti)
            opins.append(to.name)
            opis.append(to.onnx)
        opos = []
        opons = []
        for ii in range(op.OutputsLength()):
            ti = op.Outputs(ii)
            to = Tensor(im, ig, ti)
            opons.append(to.name)
            opos.append(to.onnx)

        # op - shall be per op
        assert(code.BuiltinCode() in OPTYPE_TFLITE2ONNX)
        optype = OPTYPE_TFLITE2ONNX[code.BuiltinCode()]
        n = helper.make_node(optype, opins, opons)
        onodes.append(n)

    ograph = helper.make_graph(onodes, 'init model', opis, opos)
    omodel = helper.make_model(ograph, producer_name='tflite2onnx')
    onnx.checker.check_model(omodel)
    onnx.save(omodel, onnx_path)


