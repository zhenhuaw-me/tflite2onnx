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

DTYPE_ONNX2NAME = {
    TensorProto.BOOL     :  'bool'   ,  # noqa: E203
    TensorProto.FLOAT16  :  'float16',  # noqa: E203
    TensorProto.FLOAT    :  'float32',  # noqa: E203
    TensorProto.INT16    :  'int16'  ,  # noqa: E203
    TensorProto.INT32    :  'int32'  ,  # noqa: E203
    TensorProto.INT64    :  'int64'  ,  # noqa: E203
    TensorProto.INT8     :  'int8'   ,  # noqa: E203
    TensorProto.UINT8    :  'uint8'  ,  # noqa: E203
}

DTYPE_NAME2ONNX = {
    'bool'     :  TensorProto.BOOL   ,  # noqa: E203
    'float16'  :  TensorProto.FLOAT16,  # noqa: E203
    'float32'  :  TensorProto.FLOAT  ,  # noqa: E203
    'int16'    :  TensorProto.INT16  ,  # noqa: E203
    'int32'    :  TensorProto.INT32  ,  # noqa: E203
    'int64'    :  TensorProto.INT64  ,  # noqa: E203
    'int8'     :  TensorProto.INT8   ,  # noqa: E203
    'uint8'    :  TensorProto.UINT8  ,  # noqa: E203
}


class Tensor(T2OBase):

    def __init__(self, model, graph, index, layout=None, is_initializer=False, dtype=None):
        super().__init__(model, graph, index)
        self.tflite = graph.Tensors(index) if index >= 0 else None
        self.is_initializer = is_initializer
        self.dtype = None if dtype is None else DTYPE_NAME2ONNX[dtype]
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
        if self.dtype is None:
            assert(tensor.Type() in DTYPE_TFLITE2ONNX)
            self.dtype = DTYPE_TFLITE2ONNX[tensor.Type()]
        if self.is_initializer:
            self.data = getData(self.model, self.graph, self.index, DTYPE_ONNX2NAME[self.dtype])

        self.setParsed()

    def convert(self):
        if self.status.converted:
            return
        logger.debug("Converting %s...", self.name)
        if self.is_initializer:
            self.onnx = helper.make_tensor(self.name, self.dtype, self.shape, self.data)
            onnx.checker.check_tensor(self.onnx)
        else:
            self.onnx = helper.make_tensor_value_info(self.name, self.dtype, self.shape)

        self.setConverted()

    @property
    def str(self):
        return '<' + self.name + '>' + \
               '(' + DTYPE_ONNX2NAME[self.dtype] + ',' + str(self.shape) + ')'

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
    assert(dtype in ['int32', 'float32'])
    assert(index < graph.TensorsLength())
    t = graph.Tensors(index)
    bi = t.Buffer()
    assert(bi < model.BuffersLength())
    raw = model.Buffers(bi).DataAsNumpy()
    data = np.frombuffer(raw, dtype=dtype)
    return data


def createScalar(ref, value):
    name = 'TFLITE2ONNX_Scalar_' + DTYPE_ONNX2NAME[ref.dtype] + '_' + str(value)
    if name not in registery:
        t = Tensor(ref.model, ref.graph, -1, None, True)
        t.name = name
        t.dtype = ref.dtype
        t.data = np.full((1), value, dtype=DTYPE_ONNX2NAME[ref.dtype])
        t.setParsed()
        registery[name] = t
    return registery[name]
