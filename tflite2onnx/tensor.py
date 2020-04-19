import numpy as np
import onnx
import tflite
from onnx import helper, TensorProto

from .common import BaseABC, logger
from . import layout

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

    class TFLiteObject:
        def __init__(self, model, graph, index):
            self.model = model
            self.graph = graph
            self.index = index
            self.tensor = graph.Tensors(index)

    def __init__(self, model, graph, index):
        BaseABC.__init__(self)
        self.tflite = self.TFLiteObject(model, graph, index)
        tft = self.tflite.tensor
        self.name = tft.Name().decode('utf-8')
        logger.debug("Converting %s...", self.name)
        self.shape = [int(i) for i in tft.ShapeAsNumpy()]

        assert(tft.Type() in DTYPE_MAP)
        self.dtype = DTYPE_MAP[tft.Type()]

    def create(self, isVar):
        assert(self.onnx is None)
        if isVar:
            self.onnx = helper.make_tensor_value_info(self.name, self.dtype, self.shape)
        else:
            vals = getData(self.tflite.model, self.tflite.graph, self.tflite.index, np.float32)
            self.onnx = helper.make_tensor(self.name, self.dtype, self.shape, vals)
            onnx.checker.check_tensor(self.onnx)


# The Registery holds all tensors in a SubGraph of TFLite by a name->Tensor map.
# As Registery here is *global*, we need to manually clear it when new in a SubGraph
# TODO: move the registery to Graph scope to save clear operation.
registery = {}


def getName(graph, index):
    assert(index < graph.TensorsLength())
    t = graph.Tensors(index)
    return t.Name().decode('utf-8')


def convert(model, graph, index, isVar=True):
    name = getName(graph, index)
    if name not in registery:
        t = Tensor(model, graph, index)
        t.create(isVar)
        registery[name] = t
    return registery[name]


def getData(model, graph, index, dtype):
    assert(dtype in [np.int32, np.float32])
    assert(index < graph.TensorsLength())
    t = graph.Tensors(index)
    bi = t.Buffer()
    assert(bi < model.BuffersLength())
    raw = model.Buffers(bi).DataAsNumpy()
    data = np.frombuffer(raw, dtype=dtype)
    return data


def createTransposeTensor(model, graph, index, ilayout, olayout):
    """Help to convert [NHWC -> Transpose -> NCHW -> OP -> NCHW -> Transpose -> NHWC]."""
    ref = convert(model, graph, index)
    import copy
    t = copy.copy(ref)
    t.tflite = None
    t.name = t.name + '_' + ilayout + '_to_' + olayout
    t.shape = layout.transform(t.shape, ilayout, olayout)
    t.onnx = None
    t.create(True)
    return t
