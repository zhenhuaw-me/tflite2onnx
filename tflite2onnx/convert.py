import os
import sys
import tflite
import onnx
from onnx import helper, AttributeProto, TensorProto, GraphProto
from typing import List

['BFLOAT16', 'BOOL', 'ByteSize', 'COMPLEX128', 'COMPLEX64', 'Clear', 'ClearExtension', 'ClearField', 'CopyFrom', 'DEFAULT', 'DESCRIPTOR', 'DOUBLE', 'DataLocation', 'DataType', 'DiscardUnknownFields', 'EXTERNAL', 'Extensions', 'FLOAT', 'FLOAT16', 'FindInitializationErrors', 'FromString', 'HasExtension', 'HasField', 'INT16', 'INT32', 'INT64', 'INT8', 'IsInitialized', 'ListFields', 'MergeFrom', 'MergeFromString', 'ParseFromString', 'RegisterExtension', 'STRING', 'Segment', 'SerializePartialToString', 'SerializeToString', 'SetInParent', 'UINT16', 'UINT32', 'UINT64', 'UINT8', 'UNDEFINED', 'UnknownFields', 'WhichOneof', '_CheckCalledFromGeneratedFile', '_SetListener', '__class__', '__deepcopy__', '__delattr__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__getstate__', '__gt__', '__hash__', '__init__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__setstate__', '__sizeof__', '__slots__', '__str__', '__subclasshook__', '__unicode__', '_extensions_by_name', '_extensions_by_number', 'data_location', 'data_type', 'dims', 'doc_string', 'double_data', 'external_data', 'float_data', 'int32_data', 'int64_data', 'name', 'raw_data', 'segment', 'string_data', 'uint64_data']

DTYPE_TFLITE2ONNX = {
        tflite.TensorType.BOOL    : TensorProto.BOOL   ,
        tflite.TensorType.FLOAT16 : TensorProto.FLOAT16,
        tflite.TensorType.FLOAT32 : TensorProto.FLOAT  ,
        tflite.TensorType.INT16   : TensorProto.INT16  ,
        tflite.TensorType.INT32   : TensorProto.INT32  ,
        tflite.TensorType.INT8    : TensorProto.INT8   ,
        tflite.TensorType.UINT8   : TensorProto.UINT8  ,
        }

OPTYPE_TFLITE2ONNX = {
        tflite.BuiltinOperator.ABS : 'Abs',
        }

def make_tensor(model, graph, tensor_index):
    tensor = graph.Tensors(tensor_index)
    name = tensor.Name().decode('utf-8')
    dims = [ int(i) for i in tensor.ShapeAsNumpy()]
    print(dims)
    print(type(dims))
    assert(tensor.Type() in DTYPE_TFLITE2ONNX)
    dtype = DTYPE_TFLITE2ONNX[tensor.Type()]
    # data_buf = model.Buffers(tensor.Buffer())
    # return helper.make_tensor(name, dtype, dims, data_buf, True)
    return (name, helper.make_tensor_value_info(name, dtype, dims))

def convert(tflite_path: str, onnx_path: str):
    print("tflite: %s", tflite_path)
    print("onnx: %s", onnx_path)
    with open(tflite_path, 'rb') as f:
        buf = f.read()
        im = tflite.Model.GetRootAsModel(buf, 0)

    ig = im.Subgraphs(0)

    onodes = []
    # for i in range(ig.OperatorsLength()):
    for i in range(1):
        op = ig.Operators(i)
        code = im.OperatorCodes(op.OpcodeIndex())
        assert(code.BuiltinCode() == tflite.BuiltinOperator.ABS)

        # inputs/output
        opis = []
        opins = []
        for ii in range(op.InputsLength()):
            ti = op.Inputs(ii)
            to = make_tensor(im, ig, ti)
            opins.append(to[0])
            opis.append(to[1])
        opos = []
        opons = []
        for ii in range(op.OutputsLength()):
            ti = op.Outputs(ii)
            to = make_tensor(im, ig, ti)
            opons.append(to[0])
            opos.append(to[1])

        # op - shall be per op
        assert(code.BuiltinCode() in OPTYPE_TFLITE2ONNX)
        optype = OPTYPE_TFLITE2ONNX[code.BuiltinCode()]
        n = helper.make_node(optype, opins, opons)
        onodes.append(n)

    ograph = helper.make_graph(onodes, 'init model', opis, opos)
    omodel = helper.make_model(ograph, producer_name='tflite2onnx')
    onnx.checker.check_model(omodel)
    onnx.save(omodel, onnx_path)


