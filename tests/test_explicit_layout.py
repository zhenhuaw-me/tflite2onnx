import os
import logging

import shrub
import tflite2onnx as t2o

shrub.util.formatLogging(logging.DEBUG)


def end2end_test(model_name, layout_approach, use_layout, io_layouts):
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    tflm_dir = os.path.abspath(cur_dir + '/../assets/tests')
    tflm_name = model_name + '.tflite'
    onnx_name = model_name + '.onnx'
    tflm_path = os.path.join(tflm_dir, tflm_name)
    t2o.convert(tflm_path, onnx_name, layout_approach, io_layouts)

    m = shrub.tflite.parse(tflm_path)
    m.genInput()

    onnx_ret = shrub.onnx.run(onnx_name, m.inputs, use_layout)
    tflite_ret = shrub.tflite.run(tflm_path, m.inputs)
    assert(shrub.network.cmpTensors(onnx_ret, tflite_ret))


def test_explicit_layout():
    end2end_test('abs.float32', t2o.LayoutApproach.TRANSPOSE, 'NHWC',
                 {'input':('NHWC', 'NCHW')})
    # end2end_test('abs.float32', t2o.LayoutApproach.PROPAGATION, 'NCHW',
    #              {'input':('NHWC', 'NCHW')})


if __name__ == '__main__':
    test_explicit_layout()
