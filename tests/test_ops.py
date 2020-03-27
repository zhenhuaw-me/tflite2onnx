import logging
import shrub
import tflite2onnx as t2o
import util_for_test

logging.basicConfig(format='[%(name)s:%(levelname)6s] [%(filename)12s:%(lineno)3d] [%(funcName)s] %(message)s',
                    level=logging.DEBUG)

OP_LIST = (
        'abs.float32',
        )

for op in OP_LIST:
    tflm_path = util_for_test.download(op + '.tflite')
    t2o.convert(tflm_path, op + '.onnx')

    m = shrub.tflite.parse(tflm_path)
    m.genInput()

    onnx_ret = shrub.onnx.run(op + '.onnx', m.inputs)
    tflite_ret = shrub.tflite.run(tflm_path, m.inputs)
    assert(shrub.network.cmpTensors(onnx_ret, tflite_ret))

