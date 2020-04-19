import numpy as np
import onnx
import tflite
from onnx import helper, TensorProto

from .common import BaseABC, logger

DTYPE_MAP = {
        tflite.TensorType.BOOL    : TensorProto.BOOL   ,    # noqa: E203
        tflite.TensorType.FLOAT16 : TensorProto.FLOAT16,    # noqa: E203
        tflite.TensorType.FLOAT32 : TensorProto.FLOAT  ,    # noqa: E203
        tflite.TensorType.INT16   : TensorProto.INT16  ,    # noqa: E203
        tflite.TensorType.INT32   : TensorProto.INT32  ,    # noqa: E203
        tflite.TensorType.INT8    : TensorProto.INT8   ,    # noqa: E203
        tflite.TensorType.UINT8   : TensorProto.UINT8  ,    # noqa: E203
}  # yapf: disable


class Tensor(BaseABC):

    def __init__(self, model, graph, index, isVar=True):
        self.tflite = graph.Tensors(index)
        self.name = self.tflite.Name().decode('utf-8')
        logger.debug("Converting %s...", self.name)
        self.dims = [int(i) for i in self.tflite.ShapeAsNumpy()]

        assert(self.tflite.Type() in DTYPE_MAP)
        self.dtype = DTYPE_MAP[self.tflite.Type()]

        logger.debug("Tensor <%s> isVariable: %s", self.name, self.tflite.IsVariable())

        if isVar:
            self.onnx = helper.make_tensor_value_info(self.name, self.dtype, self.dims)
        else:
            vals = getData(model, graph, index, np.float32)  # FIXME map dtype
            self.onnx = helper.make_tensor(self.name, self.dtype, self.dims, vals)
            onnx.checker.check_tensor(self.onnx)


# The Registery holds all tensors in a SubGraph of TFLite
# As Registery here is *global*, we need to manually clear it when new in a SubGraph
# TODO: move the registery to Graph scope to save clear operation.
Registery = {}


def convert(model, graph, index, isVar=True):
    if index not in Registery:
        Registery[index] = Tensor(model, graph, index, isVar)
    return Registery[index]


def createTransposeTensor(model, graph, index, ilayout, olayout):
    """Help to convert [NHWC -> Transpose -> NCHW -> OP -> NCHW -> Transpose -> NHWC]."""
    ref = convert(model, graph, index)
    import copy
    t = copy.copy(ref)
    t.tflite = None
    t.name = t.name + '_' + ilayout + '_to_' + olayout
    t.dims = transform(t.dims, ilayout, olayout)
    t.onnx = helper.make_tensor_value_info(t.name, t.dtype, t.dims)
    return t


def transform(input, ilayout: str, olayout: str):
    if (ilayout == olayout):
        return input

    perm = getPerm(ilayout, olayout)
    transfrom_axis = [input[p] for p in perm]
    return transfrom_axis


def getData(model, graph, index, dtype):
    assert(dtype in [np.int32, np.float32])
    assert(index < graph.TensorsLength())
    t = graph.Tensors(index)
    bi = t.Buffer()
    assert(bi < model.BuffersLength())
    raw = model.Buffers(bi).DataAsNumpy()
    data = np.frombuffer(raw, dtype=dtype)
    return data


def getPerm(ilayout: str, olayout: str):
    char2index = {}
    for i in range(len(ilayout)):
        c = ilayout[i]
        char2index[c] = i

    perm = [char2index[c] for c in olayout]
    return perm
