import logging

import shrub
import tflite
from tflite2onnx.op.padding import computePaddingSize

shrub.util.formatLogging(logging.DEBUG)


def test_same_trival():
    input_size = [10, 10]
    kernel_size = [3, 3]
    stride = [1, 1]
    dilation = [1, 1]
    padding_mode = tflite.Padding.SAME
    computed = computePaddingSize(padding_mode, input_size, kernel_size, stride, dilation)
    assert((computed == [1, 1, 1, 1]).all())


if __name__ == '__main__':
    test_same_trival()
