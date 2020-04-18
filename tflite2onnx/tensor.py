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

    def __init__(self, model, graph, index, transform_to_nchw):
        self.tflite = graph.Tensors(index)
        self.name = self.tflite.Name().decode('utf-8')
        logger.debug("Converting %s...", self.name)
        self.dims = [int(i) for i in self.tflite.ShapeAsNumpy()]

        # if len(self.dims) == 4 and transform_to_nchw:
        #     self.dims = transform(self.dims, 'NHWC', 'NCHW')

        assert(self.tflite.Type() in DTYPE_MAP)
        dtype = DTYPE_MAP[self.tflite.Type()]
        # data_buf = model.Buffers(self.tflite.Buffer())
        # return helper.make_tensor(self.name, dtype, self.dims, data_buf, True)

        self.onnx = helper.make_tensor_value_info(self.name, dtype, self.dims)
        # onnx.checker.check_tensor(self.onnx)


TensorMapping = {}


def create_tensor(model, graph, index, transform_to_nchw = True):
    if index not in TensorMapping:
        TensorMapping[index] = Tensor(model, graph, index, transform_to_nchw)
    return TensorMapping[index]

def transform(input, ilayout: str, olayout: str):
    if (ilayout == olayout):
        return input

    char2index = {}
    for i in range(len(ilayout)):
        c = ilayout[i]
        char2index[c] = i

    assert(isinstance(input, (list, tuple)))
    transfrom_axis = [input[char2index[c]] for c in olayout]
    return transfrom_axis

