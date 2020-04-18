import logging
import shrub
import tflite2onnx as t2o

shrub.util.formatLogging(logging.DEBUG)


OP_LIST = (
        'abs.float32',
        'add.float32',
        # 'avgpooling.float32',
        'transpose.float32',
        'softmax.float32',
        )


def test_ops():
    for op in OP_LIST:
        tflm_path = shrub.testing.download(op + '.tflite')
        t2o.convert(tflm_path, op + '.onnx')

        m = shrub.tflite.parse(tflm_path)
        m.genInput()

        onnx_ret = shrub.onnx.run(op + '.onnx', m.inputs)
        tflite_ret = shrub.tflite.run(tflm_path, m.inputs)
        assert(shrub.network.cmpTensors(onnx_ret, tflite_ret))



def test_transform():
    from tflite2onnx.tensor import transform
    assert(transform([1, 2, 6, 8], 'NCHW', 'NHWC') == [1, 6, 8, 2])
    assert(transform([1, 2, 6, 8], 'NHWC', 'NCHW') == [1, 8, 2, 6])

if __name__ == '__main__':
    test_ops()
    test_transform()
