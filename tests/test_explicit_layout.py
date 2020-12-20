import os
import logging

import shrub
import tflite2onnx as t2o

shrub.util.formatLogging(logging.DEBUG)


def run_end2end_test(tflite_path, onnx_path, tflite_layout, onnx_layout, tensors):
    io_layouts = {t: (tflite_layout, onnx_layout) for t in tensors}
    t2o.convert(tflite_path, onnx_path, io_layouts)

    m = shrub.tflite.parse(tflite_path, tflite_layout)
    m.genInput()

    onnx_ret = shrub.onnx.run(onnx_path, m.inputs, onnx_layout)
    tflite_ret = shrub.tflite.run(tflite_path, m.inputs, tflite_layout)
    assert(shrub.network.cmpTensors(onnx_ret, tflite_ret, useLayout=tflite_layout))


def end2end_test(model_name, tflite_layout, onnx_layout, tensors):
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    tflm_dir = os.path.abspath(cur_dir + '/../assets/tests')
    tflm_name = model_name + '.tflite'
    onnx_name = model_name + '.onnx'
    tflm_path = os.path.join(tflm_dir, tflm_name)

    # Firstly same ONNX layout as TFLite
    run_end2end_test(tflm_path, onnx_name, tflite_layout, tflite_layout, tensors)
    # Secondly different layouts
    run_end2end_test(tflm_path, onnx_name, tflite_layout, onnx_layout, tensors)


def test_explicit_layout():
    end2end_test('abs.float32', 'NHWC', 'NCHW', ['input', 'output'])
    end2end_test('abs.float32', 'NHWC', 'NCHW', ['input', ])

    end2end_test('add.float32', 'NHWC', 'NCHW', ['A', ])
    end2end_test('add-broadcast.float32', 'NHWC', 'NCHW', ['A', ])
    end2end_test('add-broadcast2.float32', 'NHWC', 'NCHW', ['A', ])

    end2end_test('concat.float32', 'NHWDC', 'NDCHW', ['a', ])
    end2end_test('concat2.float32', 'NHWC', 'NCHW', ['a', ])

    end2end_test('mean.float32', 'NHWC', 'NCHW', ['input', ])

    end2end_test('stridedslice.float32', 'NHWC', 'NCHW', ['input', ])
    end2end_test('stridedslice-beginmask.float32', 'NHWC', 'NCHW', ['input', ])
    end2end_test('stridedslice-endmask.float32', 'NHWC', 'NCHW', ['input', ])
    end2end_test('stridedslice-stride.float32', 'NHWC', 'NCHW', ['input', ])

    end2end_test('padding.float32', 'NHWC', 'NCHW', ['input', ])

    end2end_test('abs-sqrt.float32', 'NHWC', 'NCHW', ['input', 'output'])
    end2end_test('abs-sqrt.float32', 'NHWC', 'NCHW', ['input', ])


if __name__ == '__main__':
    test_explicit_layout()
