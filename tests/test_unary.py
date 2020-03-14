import logging
import tflite2onnx as t2o
import util_for_test

logging.basicConfig(format='[%(name)s:%(levelname)6s] [%(filename)12s:%(lineno)3d] [%(funcName)s] %(message)s',
                    level=logging.DEBUG)

OP_LIST = (
        'abs.int32',
        )

for op in OP_LIST:
    tflm_path = util_for_test.getTFLiteModel(op)
    t2o.convert(tflm_path, op + '.onnx')
