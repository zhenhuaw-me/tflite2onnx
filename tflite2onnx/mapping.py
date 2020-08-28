from onnx import TensorProto
from tflite import TensorType


def inverseDict(d):
    return {v: k for k, v in d.items()}


def buildIndirectMapping(a, b):
    """Given a maps x->y, b maps y->z, return map of x->z."""
    assert(len(a) == len(b))
    assert(isinstance(list(b.keys())[0], type(list(a.values())[0])))
    c = dict()
    for x in a.keys():
        y = a[x]
        z = b[y]
        c[x] = z
    return c


DTYPE_ONNX2NAME = {
    TensorProto.BOOL: 'bool',
    TensorProto.FLOAT16: 'float16',
    TensorProto.FLOAT: 'float32',
    TensorProto.INT16: 'int16',
    TensorProto.INT32: 'int32',
    TensorProto.INT64: 'int64',
    TensorProto.INT8: 'int8',
    TensorProto.UINT8: 'uint8',
}

DTYPE_NAME2ONNX = inverseDict(DTYPE_ONNX2NAME)

DTYPE_TFLITE2NAME = {
    TensorType.BOOL: 'bool',
    TensorType.FLOAT16: 'float16',
    TensorType.FLOAT32: 'float32',
    TensorType.INT16: 'int16',
    TensorType.INT32: 'int32',
    TensorType.INT64: 'int64',
    TensorType.INT8: 'int8',
    TensorType.UINT8: 'uint8',
}

DTYPE_NAME2TFLITE = inverseDict(DTYPE_TFLITE2NAME)

DTYPE_TFLITE2ONNX = buildIndirectMapping(DTYPE_TFLITE2NAME, DTYPE_NAME2ONNX)
DTYPE_ONNX2TFLITE = buildIndirectMapping(DTYPE_ONNX2NAME, DTYPE_NAME2TFLITE)
