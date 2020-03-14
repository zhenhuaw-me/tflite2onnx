import tflite
from onnx import helper, TensorProto

DTYPE_TFLITE2ONNX = {
        tflite.TensorType.BOOL    : TensorProto.BOOL   ,
        tflite.TensorType.FLOAT16 : TensorProto.FLOAT16,
        tflite.TensorType.FLOAT32 : TensorProto.FLOAT  ,
        tflite.TensorType.INT16   : TensorProto.INT16  ,
        tflite.TensorType.INT32   : TensorProto.INT32  ,
        tflite.TensorType.INT8    : TensorProto.INT8   ,
        tflite.TensorType.UINT8   : TensorProto.UINT8  ,
        }

class Tensor(object):
    name = 'Uninitialized'
    tflite = None
    onnx = None
    def __init__(self, model, graph, tfl_index):
        self.tflite = graph.Tensors(tfl_index)
        self.name = self.tflite.Name().decode('utf-8')
        dims = [ int(i) for i in self.tflite.ShapeAsNumpy()]
        assert(self.tflite.Type() in DTYPE_TFLITE2ONNX)
        dtype = DTYPE_TFLITE2ONNX[self.tflite.Type()]
        # data_buf = model.Buffers(self.tflite.Buffer())
        # return helper.make_tensor(self.name, dtype, dims, data_buf, True)
        self.onnx = helper.make_tensor_value_info(self.name, dtype, dims)

