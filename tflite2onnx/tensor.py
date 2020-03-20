import tflite
import onnx
from onnx import helper, TensorProto

from .common import BaseABC, logger

DTYPE_MAP = {
        tflite.TensorType.BOOL    : TensorProto.BOOL   ,
        tflite.TensorType.FLOAT16 : TensorProto.FLOAT16,
        tflite.TensorType.FLOAT32 : TensorProto.FLOAT  ,
        tflite.TensorType.INT16   : TensorProto.INT16  ,
        tflite.TensorType.INT32   : TensorProto.INT32  ,
        tflite.TensorType.INT8    : TensorProto.INT8   ,
        tflite.TensorType.UINT8   : TensorProto.UINT8  ,
        }

class Tensor(BaseABC):

    def __init__(self, model, graph, index):
        self.tflite = graph.Tensors(index)
        self.name = self.tflite.Name().decode('utf-8')
        logger.debug("[Tensor] Converting {}...".format(self.name))
        dims = [ int(i) for i in self.tflite.ShapeAsNumpy()]
        assert(self.tflite.Type() in DTYPE_MAP)
        dtype = DTYPE_MAP[self.tflite.Type()]
        # data_buf = model.Buffers(self.tflite.Buffer())
        # return helper.make_tensor(self.name, dtype, dims, data_buf, True)

        self.onnx = helper.make_tensor_value_info(self.name, dtype, dims)
        print(type(self.onnx))
        # onnx.checker.check_tensor(self.onnx)

    # def __str__(self):
    #     return helper.


TensorMapping = {}

def create_tensor(model, graph, index):
    if index not in TensorMapping:
        TensorMapping[index] = Tensor(model, graph, index)
    return TensorMapping[index]

