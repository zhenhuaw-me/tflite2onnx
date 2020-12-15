import copy
import logging
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from tflite2onnx import mapping
from tflite2onnx.common import T2OBase

logger = logging.getLogger('tflite2onnx')


class Tensor(T2OBase):
    def __init__(self, model, graph, index, layout=None, is_bias=False):
        super().__init__(model, graph, index)
        self.tflite = graph.Tensors(index) if index >= 0 else None
        self.shape = []
        self.dtype = None
        self.data = None

        # the defaults of quantization parameter
        self.scale = 1.0
        self.zero_point = 127

        self.layout = layout
        self.producers = []
        self.consumers = []

        # we only accept INT32 as quantized tensor type for bias
        self.is_bias = is_bias

        self.setInited()

    @property
    def isInitializer(self):
        return self.data is not None

    def addProducer(self, op):
        assert(len(self.producers) == 0)
        self.producers.append(op)
        assert(len(self.producers) == 1)

    def removeProducer(self, op):
        assert(len(self.producers) == 1)
        assert(self.producers[0] == op)
        self.producers.remove(op)

    def replaceProducer(self, original, new):
        assert(len(self.producers) == 1)
        assert(self.producers[0] == original)
        self.producers[0] = new

    def addConsumer(self, op):
        assert(op not in self.consumers)
        self.consumers.append(op)

    def removeConsumer(self, op):
        assert(op in self.consumers)
        self.consumers.remove(op)

    def replaceConsumer(self, original, new):
        assert(original in self.consumers)
        for i, op in enumerate(self.consumers):
            if op is original:
                self.consumers[i] = new
                return

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
        logger.debug("Dequantizing %s", self.shorty)
        if self.isInitializer:
            int32 = self.data.astype('int32')
            shiftted = np.subtract(int32, self.zero_point)
            fp32 = np.multiply(shiftted.astype('float32'), self.scale)
            self.data = fp32
        self.dtype = TensorProto.FLOAT

    @property
    def isScalar(self):
        return (self.layout is None) and (len(self.shape) == 0) and (len(self.data) == 1)

    def asDtype(self, dtype: str):
        self.dtype = mapping.DTYPE_NAME2ONNX[dtype]
        if self.isInitializer:
            self.data = self.data.astype(dtype)

    def parse(self):
        if self.status.parsed:
            return
        tensor = self.tflite
        self.name = tensor.Name().decode('utf-8')
        logger.debug("Parsing %s...", self.name)
        self.shape = [int(i) for i in tensor.ShapeAsNumpy()]

        assert(tensor.Type() in mapping.DTYPE_TFLITE2ONNX)
        self.dtype = mapping.DTYPE_TFLITE2ONNX[tensor.Type()]

        if self.quantized:
            quant = tensor.Quantization()
            assert(quant.ScaleAsNumpy().size == 1), "Per-tensor support only currently"
            assert(quant.ZeroPointAsNumpy().size == 1), "Per-tensor support only currently"
            self.scale = float(quant.ScaleAsNumpy()[0])
            self.zero_point = int(quant.ZeroPointAsNumpy()[0])

        self.data = TensorFactory.getData(self.model, self.graph, self.index,
                                          mapping.DTYPE_ONNX2NAME[self.dtype])

        self.setParsed()

    def transform(self):
        assert(self.status.parsed)
        assert(self.layout is not None)
        if self.isInitializer:
            data = self.data.reshape(self.shape)
            self.shape = self.layout.transform(self.shape)
            self.data = data.transpose(self.layout.perm)
        else:
            self.shape = self.layout.transform(self.shape)

    def validate(self):
        if self.isInitializer:
            assert(len(self.producers) == 0), "Initializer should not have producer"
        else:
            assert(len(self.producers) <= 1), "Tensor should have 1 producer or no"
        assert(len(self.name) > 0), "Tensor must have valid name"

    def convert(self):
        if self.status.converted:
            return
        logger.debug("Converting %s...", self.shorty)
        if self.isInitializer:
            if isinstance(self.data, np.ndarray):
                # Need this because ONNX saves non-C-builtin data type in a special way.
                # https://github.com/onnx/onnx/blob/v1.8.0/onnx/onnx.proto3#L523
                self.onnx = numpy_helper.from_array(self.data, self.name)
            else:
                self.onnx = helper.make_tensor(self.name, self.dtype, self.shape, self.data)
            onnx.checker.check_tensor(self.onnx)
        else:
            self.onnx = helper.make_tensor_value_info(self.name, self.dtype, self.shape)
            onnx.checker.check_value_info(self.onnx)
        assert(self.onnx)

        self.setConverted()

    @property
    def shorty(self):
        return '<%s>(%s,%s)' % (self.name, mapping.DTYPE_ONNX2NAME[self.dtype], self.shape)

    def __str__(self):
        producer_names = [op.shorty for op in self.producers]
        consumer_names = [op.shorty for op in self.consumers]
        return '%s: {%s} -> {%s}' % (self.shorty, producer_names, consumer_names)


class TensorFactory:
    """The Registery holds all tensors in a SubGraph of TFLite by a name->Tensor map."""
    def __init__(self, model, graph):
        self.model = model
        self.graph = graph
        self.registery = dict()

    def get(self, index, layout=None, is_bias=False):
        tft = self.graph.Tensors(index)
        name = tft.Name().decode('utf-8')
        if name not in self.registery:
            t = Tensor(self.model, self.graph, index, layout, is_bias)
            self.registery[name] = t
        else:
            t = self.registery[name]
            if t.layout is None:
                t.layout = layout
        return t

    def getWithRef(self, ref, name, forceUnique=False):
        """Create a copy of the ref tensor.

        This is used to create helper tensors for activations, layout handling,
        quantization and so on. Some attributions will be removed.
        """
        if name not in self.registery:
            t = Tensor(self.model, self.graph, -1)
            t.name = name
            t.dtype = ref.dtype
            t.layout = copy.deepcopy(ref.layout)
            t.shape = copy.deepcopy(ref.shape)
            t.scale = copy.deepcopy(ref.scale)
            t.zero_point = copy.deepcopy(ref.zero_point)
            self.registery[name] = t
        else:
            assert(not forceUnique)
            t = self.registery[name]
        return t

    def createScalar(self, dtype, value):
        name = 'TFLITE2ONNX_Scalar_' + dtype + '_' + str(value)
        return self._createScalarCore(name, dtype, value)

    def createVector(self, ndarray):
        array2key = str(ndarray).replace(' ', '_')
        dtype = str(ndarray.dtype)
        name = 'TFLITE2ONNX_Vector_' + dtype + '_' + array2key
        if name not in self.registery:
            t = Tensor(self.model, self.graph, -1, None)
            t.name = name
            t.dtype = mapping.DTYPE_NAME2ONNX[dtype]
            t.data = ndarray.copy()
            t.shape = t.data.shape
            t.setParsed()
            self.registery[name] = t
        return self.registery[name]

    def createEmptyTensor(self):
        # Used for optional inputs that we need it to be empty.
        logger.warning("Empty tensor used, please double confirm your code path!")
        name = 'TFLITE2ONNX_EmptyTensor'
        if name not in self.registery:
            t = Tensor(self.model, self.graph, -1, None)
            t.name = name
            t.dtype = mapping.DTYPE_NAME2ONNX['float32']
            t.data = []
            t.shape = [0]
            t.setParsed()
            self.registery[name] = t
        return self.registery[name]

    def _createScalarCore(self, name, dtype, value):
        if name not in self.registery:
            t = Tensor(self.model, self.graph, -1, None)
            t.name = name
            t.dtype = mapping.DTYPE_NAME2ONNX[dtype]
            t.data = [value]  # cannot use NDArray for cases such as min/max of ReLU6
            t.setParsed()
            self.registery[name] = t
        return self.registery[name]

    def createQuantScale(self, tensor):
        value = tensor.scale
        assert(isinstance(value, float) or (len(value) == 1))
        dtype = 'float32'
        name = 'TFLITE2ONNX_Scalar_' + dtype + '_' + str(value)
        return self._createScalarCore(name, dtype, value)

    def createQuantZeroPoint(self, tensor):
        value = tensor.zero_point
        assert(isinstance(value, int) or (len(value) == 1))
        assert(value >= 0 and value <= 255)
        dtype = 'uint8'
        name = 'TFLITE2ONNX_Scalar_' + dtype + '_' + str(value)
        return self._createScalarCore(name, dtype, value)

    @staticmethod
    def getData(model, graph, index, dtype):
        if (dtype not in ['int32', 'float32', 'uint8']):
            logger.warning("Data type {} not supported/tested yet, "
                           "the generated model may contain error".format(dtype))
        assert(index < graph.TensorsLength())
        t = graph.Tensors(index)
        bi = t.Buffer()
        shape = t.ShapeAsNumpy()
        assert(bi < model.BuffersLength())
        raw = model.Buffers(bi).DataAsNumpy()
        if isinstance(raw, int) and raw == 0:
            return None
        data = np.frombuffer(raw, dtype=dtype)
        if len(shape) > 0:
            data = data.reshape(shape)
        return data.copy()
