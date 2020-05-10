import numpy as np
import onnx
import tflite
from onnx import helper, TensorProto

from .common import T2OBase, logger
from .op import Operator


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

    def __init__(self, model, graph, index, layout=None, is_initializer=False):
        super().__init__(model, graph, index)
        self.tflite = graph.Tensors(index) if index >= 0 else None
        self.is_initializer = is_initializer
        self.dtype = None
        self.shape = []
        self.data = None

        self.layout = layout
        self.producers = []
        self.consumers = []

        self.setInited()

    def addProducer(self, op):
        assert(isinstance(op, Operator))
        if op not in self.producers:
            self.producers.append(op)

    def addConsumer(self, op):
        assert(isinstance(op, Operator))
        if op not in self.consumers:
            self.consumers.append(op)

    @property
    def layoutMatch(self):
        if self.layout is None:
            return True
        else:
            return self.layout.match

    def parse(self):
        tensor = self.tflite
        self.name = tensor.Name().decode('utf-8')
        logger.debug("Parsing %s...", self.name)
        self.shape = [int(i) for i in tensor.ShapeAsNumpy()]
        assert(tensor.Type() in DTYPE_TFLITE2ONNX)
        self.dtype = DTYPE_TFLITE2ONNX[tensor.Type()]

        self.setParsed()

    def convert(self):
        if self.status.converted:
            return
        logger.debug("Converting %s...", self.name)
        if self.is_initializer:
            vals = getData(self.model, self.graph, self.index, np.float32)
            self.onnx = helper.make_tensor(self.name, self.dtype, self.shape, vals)
            onnx.checker.check_tensor(self.onnx)
        else:
            self.onnx = helper.make_tensor_value_info(self.name, self.dtype, self.shape)

        self.setConverted()

    @property
    def str(self):
        DTYPE_NAME = {
            TensorProto.BOOL     :  'bool',    # noqa: E203
            TensorProto.FLOAT16  :  'float16',    # noqa: E203
            TensorProto.FLOAT    :  'float(32)',    # noqa: E203
            TensorProto.INT16    :  'int16',    # noqa: E203
            TensorProto.INT32    :  'int32',    # noqa: E203
            TensorProto.INT8     :  'int8',    # noqa: E203
            TensorProto.UINT8    :  'uint',    # noqa: E203
            }
        return '<' + self.name + '>' + '(' + DTYPE_NAME[self.dtype] + ',' + str(self.shape) + ')'

    def __str__(self):
        producer_names = str([op.str for op in self.producers])
        consumer_names = str([op.str for op in self.consumers])
        return self.str + ': ' + producer_names + ' -> ' + consumer_names


def parseTensorName(graph, index):
    assert(index < graph.TensorsLength())
    t = graph.Tensors(index)
    return t.Name().decode('utf-8')


def get(model, graph, index, layout=None, is_initializer=False):
    name = parseTensorName(graph, index)
    if name not in registery:
        t = Tensor(model, graph, index, layout, is_initializer)
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
