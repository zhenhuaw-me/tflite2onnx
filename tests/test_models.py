import os
import logging

import shrub
import tflite2onnx as t2o

shrub.util.formatLogging(logging.DEBUG)


def end2end_test(model_name, layout_approach, use_layout):
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    tflm_dir = os.path.abspath(cur_dir + '/../assets/tests')
    tflm_name = model_name + '.tflite'
    onnx_name = model_name + '.onnx'
    tflm_path = os.path.join(tflm_dir, tflm_name)
    t2o.convert(tflm_path, onnx_name, layout_approach)

    m = shrub.tflite.parse(tflm_path)
    m.genInput()

    onnx_ret = shrub.onnx.run(onnx_name, m.inputs, use_layout)
    tflite_ret = shrub.tflite.run(tflm_path, m.inputs)
    assert(shrub.network.cmpTensors(onnx_ret, tflite_ret))


def test_ops_implicit_layout():
    OP_LIST_IMPLICIT_LAYOUT = (
        'avgpooling.float32',
        'avgpool-concat.float32',
        'conv.float32',
        'conv-dilation.float32',
        'conv-relu6.float32',
        'conv-stride.float32',
        'depthwise-conv.float32',
        'depthwise-conv-stride.float32',
    )

    for op in OP_LIST_IMPLICIT_LAYOUT:
        end2end_test(op, t2o.LayoutApproach.TRANSPOSE, 'NHWC')
        end2end_test(op, t2o.LayoutApproach.PROPAGATION, 'NCHW')


def test_ops_layout_transparent():
    OP_LIST_LAYOUT_TRANSPARENT = (
        'abs.float32',
        'add.float32',
        'concat.float32',
        'relu6.float32',
        'relu.float32',
        'reshape.float32',
        'softmax.float32',
        'transpose.float32',
    )

    for op in OP_LIST_LAYOUT_TRANSPARENT:
        end2end_test(op, t2o.LayoutApproach.DEFAULT, 'NHWC')


def test_networks():
    NETWORK_LIST = (
        'mobilenet_v1_0.25_128',
    )

    for net in NETWORK_LIST:
        end2end_test(net, t2o.LayoutApproach.TRANSPOSE, 'NHWC')
        end2end_test(net, t2o.LayoutApproach.PROPAGATION, 'NCHW')


if __name__ == '__main__':
    test_ops_implicit_layout()
    test_ops_layout_transparent()
    test_networks()
