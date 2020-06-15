import os
import logging

import shrub
import tflite2onnx as t2o

shrub.util.formatLogging(logging.DEBUG)


def end2end_test(model_name):
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    tflm_dir = os.path.abspath(cur_dir + '/../assets/tests')
    tflm_name = model_name + '.tflite'
    onnx_name = model_name + '.onnx'
    tflm_path = os.path.join(tflm_dir, tflm_name)
    t2o.convert(tflm_path, onnx_name)

    m = shrub.tflite.parse(tflm_path)
    m.genInput()

    onnx_ret = shrub.onnx.run(onnx_name, m.inputs, layout='NHWC')
    tflite_ret = shrub.tflite.run(tflm_path, m.inputs)
    assert(shrub.network.cmpTensors(onnx_ret, tflite_ret))


def test_ops():
    OP_LIST = (
        'abs.float32',
        'add.float32',
        'avgpooling.float32',
        'avgpool-concat.float32',
        'concat.float32',
        'conv.float32',
        'conv-relu6.float32',
        'conv-stride.float32',
        'depthwise-conv.float32',
        'depthwise-conv-stride.float32',
        'relu6.float32',
        'relu.float32',
        'reshape.float32',
        'softmax.float32',
        'transpose.float32',
    )

    for op in OP_LIST:
        end2end_test(op)


def test_networks():
    NETWORK_LIST = (
        'mobilenet_v1_0.25_128',
    )

    for net in NETWORK_LIST:
        end2end_test(net)


if __name__ == '__main__':
    test_ops()
    # test_networks()
