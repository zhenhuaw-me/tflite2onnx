import os
import logging

import shrub
import tflite2onnx as t2o

shrub.util.formatLogging(logging.DEBUG)


def end2end_test(model_name, use_layout, io_layouts):
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    tflm_dir = os.path.abspath(cur_dir + '/../assets/tests')
    tflm_name = model_name + '.tflite'
    onnx_name = model_name + '.onnx'
    tflm_path = os.path.join(tflm_dir, tflm_name)
    t2o.convert(tflm_path, onnx_name, io_layouts)

    m = shrub.tflite.parse(tflm_path)
    m.genInput()

    onnx_ret = shrub.onnx.run(onnx_name, m.inputs, use_layout)
    tflite_ret = shrub.tflite.run(tflm_path, m.inputs)
    assert(shrub.network.cmpTensors(onnx_ret, tflite_ret))


def test_explicit_layout():
    end2end_test('abs.float32', 'NCHW', {'input': ('NHWC', 'NCHW'), 'output': ('NHWC', 'NCHW')})
    end2end_test('abs.float32', 'NCHW', {'input': ('NHWC', 'NCHW')})
    end2end_test('abs.float32', 'NHWC', {'input': ('NHWC', 'NHWC'), 'output': ('NHWC', 'NHWC')})
    end2end_test('abs.float32', 'NHWC', {'input': ('NHWC', 'NHWC')})

    end2end_test('add-broadcast.float32', 'NHWC', {'A': ('NHWC', 'NHWC')})
    end2end_test('add-broadcast.float32', 'NCHW', {'A': ('NHWC', 'NCHW')})
    end2end_test('add-broadcast.float32', 'NHWC', {'output': ('NHWC', 'NHWC')})
    end2end_test('add-broadcast.float32', 'NCHW', {'output': ('NHWC', 'NCHW')})
    end2end_test('add-broadcast.float32', 'NCHW', {'A': ('NHWC', 'NCHW'),
                                                   'output': ('NHWC', 'NCHW')})

    end2end_test('add-broadcast2.float32', 'NHWC', {'A': ('NHWC', 'NHWC')})
    end2end_test('add-broadcast2.float32', 'NCHW', {'A': ('NHWC', 'NCHW')})
    end2end_test('add-broadcast2.float32', 'NHWC', {'output': ('NHWC', 'NHWC')})
    end2end_test('add-broadcast2.float32', 'NCHW', {'output': ('NHWC', 'NCHW')})
    end2end_test('add-broadcast2.float32', 'NCHW', {'A': ('NHWC', 'NCHW'),
                                                    'output': ('NHWC', 'NCHW')})

    end2end_test('concat2.float32', 'NHWC', {'a': ('NHWC', 'NHWC')})
    end2end_test('concat2.float32', 'NCHW', {'a': ('NHWC', 'NCHW')})

    end2end_test('mean.float32', 'NHWC', {'input': ('NHWC', 'NHWC')})
    end2end_test('mean.float32', 'NCHW', {'input': ('NHWC', 'NCHW')})

    end2end_test('stridedslice.float32', 'NHWC', {'input': ('NHWC', 'NHWC')})
    end2end_test('stridedslice.float32', 'NCHW', {'input': ('NHWC', 'NCHW')})
    end2end_test('stridedslice-beginmask.float32', 'NHWC', {'input': ('NHWC', 'NHWC')})
    end2end_test('stridedslice-beginmask.float32', 'NCHW', {'input': ('NHWC', 'NCHW')})
    end2end_test('stridedslice-endmask.float32', 'NHWC', {'input': ('NHWC', 'NHWC')})
    end2end_test('stridedslice-endmask.float32', 'NCHW', {'input': ('NHWC', 'NCHW')})
    end2end_test('stridedslice-stride.float32', 'NHWC', {'input': ('NHWC', 'NHWC')})
    end2end_test('stridedslice-stride.float32', 'NCHW', {'input': ('NHWC', 'NCHW')})

    end2end_test('padding.float32', 'NHWC', {'input': ('NHWC', 'NHWC')})
    end2end_test('padding.float32', 'NCHW', {'input': ('NHWC', 'NCHW')})


if __name__ == '__main__':
    test_explicit_layout()
