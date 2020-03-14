import tflite2onnx as t2o
import util_for_test

OP_LIST = (
        'abs.int32',
        )

for op in OP_LIST:
    tflm_path = util_for_test.getTFLiteModel(op)
    t2o.convert(tflm_path, op + '.onnx')
