import numpy as np
import onnx
import tflite
from onnx import helper, TensorProto

from .common import T2OBase, logger
from . import layout


# The Registery holds all tensors in a SubGraph of TFLite by a name->Tensor map.
# As Registery here is *global*, we need to manually clear it when new in a SubGraph
# TODO: move the registery to Graph scope to save clear operation.
registery = {}


DTYPE_TFLITE2ONNX = {
        tflite.TensorType.BOOL    : TensorProto.BOOL   ,    # noqa: E203
        tflite.TensorType.FLOAT16 : TensorProto.FLOAT16,    # noqa: E203
        tflite.TensorType.FLOAT32 : TensorProto.FLOAT  ,    # noqa: E203
        tflite.TensorType.INT16   : TensorProto.INT16  ,    # noqa: E203
        tflite.TensorType.INT32   : TensorProto.INT32  ,    # noqa: E203
        tflite.TensorType.INT8    : TensorProto.INT8   ,    # noqa: E203
        tflite.TensorType.UINT8   : TensorProto.UINT8  ,    # noqa: E203
}  # yapf: disable


class Tensor(T2OBase):

    def __init__(self, model, graph, index):
        super().__init__(model, graph, index)
        self.tflite = graph.Tensors(index)
        self.setInited()

    def parse(self):
        self.name = self.tflite.Name().decode('utf-8')
        logger.debug("Parsing %s...", self.name)
        self.shape = [int(i) for i in self.tflite.ShapeAsNumpy()]
        assert(self.tflite.Type() in DTYPE_TFLITE2ONNX)
        self.dtype = DTYPE_TFLITE2ONNX[self.tflite.Type()]
        self.setParsed()

    def buildGraph(self):
        logger.debug("Building graph...")
        self.setGraphBuilt()

    def propagate(self):
        logger.debug("Propagating...")
        # TODO: handle layout issue.
        self.setPropagated()

    def convert(self, isVar=True):
        logger.debug("Converting %s...", self.name)
        assert(self.onnx is None)
        if isVar:
            self.onnx = helper.make_tensor_value_info(self.name, self.dtype, self.shape)
        else:
            vals = getData(self.model, self.graph, self.index, np.float32)
            self.onnx = helper.make_tensor(self.name, self.dtype, self.shape, vals)
            onnx.checker.check_tensor(self.onnx)


def getName(graph, index):
    assert(index < graph.TensorsLength())
    t = graph.Tensors(index)
    return t.Name().decode('utf-8')


def get(model, graph, index, isVar=True):
    name = getName(graph, index)
    if name not in registery:
        t = Tensor(model, graph, index)
        # t.parse()
        # t.create(isVar)
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
    ref = get(model, graph, index)
    import copy
    t = copy.copy(ref)
    t.tflite = None
    t.name = t.name + '_' + ilayout + '_to_' + olayout
    t.shape = layout.transform(t.shape, ilayout, olayout)
    t.onnx = None
    t.create(True)
    return t
