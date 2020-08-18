import logging
import numpy as np
import onnx
import tflite
from onnx import helper, TensorProto
from shrub.mapping import DTYPE_TFLITE2ONNX, DTYPE_NAME2ONNX, DTYPE_ONNX2NAME

from tflite2onnx.common import T2OBase
from tflite2onnx.op import Operator

logger = logging.getLogger('tflite2onnx')

# The Registery holds all tensors in a SubGraph of TFLite by a name->Tensor map.
# As Registery here is *global*, we need to manually clear it when new in a SubGraph
# TODO: move the registery to Graph scope to save clear operation.
registery = {}


class Tensor(T2OBase):
    def __init__(self, model, graph, index, layout=None,
                 is_initializer=False, is_bias=False):
        super().__init__(model, graph, index)
        self.tflite = graph.Tensors(index) if index >= 0 else None
        self.is_initializer = is_initializer
        self.is_bias = is_bias
        self.shape = []
        self.dtype = None

        # the defaults of quantization parameter
        self.scale = 1.0
        self.zero_point = 127
        self.data = None

        self.layout = layout
        self.producers = []
        self.consumers = []

        self.setInited()

    def addProducer(self, op):
        assert(isinstance(op, Operator))
        if op not in self.producers:
            self.producers.append(op)

    def removeProducer(self, op):
        assert(isinstance(op, Operator))
        if op in self.producers:
            self.producers.remove(op)

    def replaceProducer(self, original, new):
        assert(len(self.producers) == 1)
        assert(isinstance(original, Operator))
        assert(isinstance(new, Operator))
        assert(self.producers[0] == original)
        self.producers[0] = new

    def addConsumer(self, op):
        assert(isinstance(op, Operator))
        if op not in self.consumers:
            self.consumers.append(op)

    def removeConsumer(self, op):
        assert(isinstance(op, Operator))
        if op in self.consumers:
            self.consumers.remove(op)

    def replaceConsumer(self, original, new):
        assert(isinstance(original, Operator))
        assert(isinstance(new, Operator))
        for index, op in enumerate(self.consumers):
            if op is original:
                self.consumers[index] = new
                return
        assert(False)

    @property
    def quantized(self):
        is_quant_dtype = ((self.dtype == TensorProto.UINT8) or
                          ((self.dtype == TensorProto.INT32) and self.is_bias))
        if self.tflite is None:
            return is_quant_dtype
        else:
            has_quant = self.tflite.Quantization() is not None
            return is_quant_dtype and has_quant

    def dequantize(self):
        if not self.quantized:
            return
        logger.debug("Dequantizing %s", self.name)
        if self.is_initializer:
            self.data = (self.data - self.zero_point) * self.scale
        self.dtype = TensorProto.FLOAT

    @property
    def layoutMatch(self):
        if self.layout is None:
            return True
        else:
            return self.layout.match

    @property
    def isScalar(self):
        return (self.layout is None) and (len(self.shape) == 0) and (len(self.data) == 1)

    def parse(self):
        if self.status.parsed:
            return
        tensor = self.tflite
        self.name = tensor.Name().decode('utf-8')
        logger.debug("Parsing %s...", self.name)
        self.shape = [int(i) for i in tensor.ShapeAsNumpy()]

        assert(tensor.Type() in DTYPE_TFLITE2ONNX)
        assert(self.dtype is None)
        self.dtype = DTYPE_TFLITE2ONNX[tensor.Type()]

        if self.quantized:
            quant = tensor.Quantization()
            assert(quant.ScaleAsNumpy().size == 1), "Per-tensor support only currently"
            assert(quant.ZeroPointAsNumpy().size == 1), "Per-tensor support only currently"
            self.scale = float(quant.ScaleAsNumpy()[0])
            self.zero_point = int(quant.ZeroPointAsNumpy()[0])

        if self.is_initializer:
            self.data = getData(self.model, self.graph, self.index, DTYPE_ONNX2NAME[self.dtype])

        self.setParsed()

    def transform(self):
        assert(self.status.parsed)
        assert(self.layout is not None)
        if self.is_initializer:
            data = self.data.reshape(self.shape)
            self.shape = self.layout.transform(self.shape)
            data = data.transpose(self.layout.perm)
            self.data = data.flatten()
        else:
            self.shape = self.layout.transform(self.shape)

    def convert(self):
        if self.status.converted:
            return
        logger.debug("Converting %s...", self.name)
        if self.is_initializer:
            self.onnx = helper.make_tensor(self.name, self.dtype, self.shape, self.data)
            onnx.checker.check_tensor(self.onnx)
        else:
            self.onnx = helper.make_tensor_value_info(self.name, self.dtype, self.shape)
        assert(self.onnx)

        self.setConverted()

    @property
    def str(self):
        return '<%s>(%s,%s)' % (self.name, DTYPE_ONNX2NAME[self.dtype], str(self.shape))

    def __str__(self):
        producer_names = str([op.str for op in self.producers])
        consumer_names = str([op.str for op in self.consumers])
        return '%s: %s -> %s' % (self.str, producer_names, consumer_names)


def get(model, graph, index, layout=None, is_initializer=False, is_bias=False):
    tft = graph.Tensors(index)
    name = tft.Name().decode('utf-8')
    if name not in registery:
        t = Tensor(model, graph, index, layout, is_initializer, is_bias)
        registery[name] = t

    # We need to handle scenario that, `producer.implicitLayout` is `False`, while
    # `consumer.implicitLayout` is `True`. Assigning new `layout` to Tensor cannot
    # get it fixed, as the *transpose approach* may break the semantic of producer.
    # Needs further consideration.
    # else:
    #     t = registery[name]
    #     if layout is not None:
    #         if t.layout is None:
    #             # In case the producer doesn't assume implicit layout, but a consumer does.
    #             t.layout = layout
    #         else:
    #             assert(t.layout.source == layout.source)
    #             assert(t.layout.target == layout.target)

    return registery[name]


def getData(model, graph, index, dtype):
    assert(dtype in ['int32', 'float32', 'uint8'])
    assert(index < graph.TensorsLength())
    t = graph.Tensors(index)
    bi = t.Buffer()
    assert(bi < model.BuffersLength())
    raw = model.Buffers(bi).DataAsNumpy()
    data = np.frombuffer(raw, dtype=dtype)
    return data


def isTFLiteQuantized(graph, tensor_index):
    t = graph.Tensors(tensor_index)
    return ((t.Type() == tflite.TensorType.UINT8) and
            (t.Quantization() is not None))


def createScalar(ref, value):
    name = 'TFLITE2ONNX_Scalar_' + DTYPE_ONNX2NAME[ref.dtype] + '_' + str(value)
    dtype = DTYPE_ONNX2NAME[ref.dtype]
    return _createScalarCore(ref.model, ref.graph, name, dtype, value)


def _createScalarCore(model, graph, name, dtype, value):
    if name not in registery:
        t = Tensor(model, graph, -1, None, True)
        t.name = name
        t.dtype = DTYPE_NAME2ONNX[dtype]
        t.data = np.full((1), value, dtype=dtype)
        t.setParsed()
        registery[name] = t
    return registery[name]


def createQuantScale(tensor):
    value = tensor.scale
    assert(isinstance(value, float) or (len(value) == 1))
    dtype = 'float32'
    name = 'TFLITE2ONNX_Scalar_' + dtype + '_' + str(value)
    return _createScalarCore(tensor.model, tensor.graph, name, dtype, value)


def createQuantZeroPoint(tensor):
    value = tensor.zero_point
    assert(isinstance(value, int) or (len(value) == 1))
    assert(value >= 0 and value <= 255)
    dtype = 'uint8'
    name = 'TFLITE2ONNX_Scalar_' + dtype + '_' + str(value)
    return _createScalarCore(tensor.model, tensor.graph, name, dtype, value)
