import numpy as np
import tflite


PaddingMapping = {
    tflite.Padding.SAME: 'SAME_UPPER',
    tflite.Padding.VALID: 'VALID',
}


# https://github.com/tensorflow/tensorflow/blob/v2.2.0/tensorflow/lite/kernels/padding.h#L58
def computePaddingSize(padding_mode, input_size, kernel_size, stride, dilation):
    assert(len(input_size) == len(kernel_size))
    assert(len(input_size) == len(stride))
    assert(len(input_size) == len(dilation))

    # compute output shape
    ones = np.ones_like(input_size)
    effective_filter_size = np.add(np.multiply(np.subtract(kernel_size, ones), dilation), ones)
    if padding_mode is tflite.Padding.SAME:
        oshape = np.divide(np.subtract(np.add(input_size, stride), ones), stride)
    elif padding_mode is tflite.Padding.VALID:
        oshape = np.divide(np.subtract(np.add(input_size, stride), effective_filter_size), stride)
    else:
        raise ValueError("Unknown padding mode!")

    # infer the padding
    total_padding = np.add(np.multiply(np.subtract(oshape, ones), stride),
                           np.subtract(effective_filter_size, input_size))
    total_padding = np.maximum(total_padding, np.zeros_like(input_size))
    total_padding = total_padding.astype('int')

    # ONNX semantic
    pre_padding = total_padding // 2
    post_padding = np.subtract(total_padding, pre_padding)
    padding = np.concatenate((pre_padding, post_padding))

    return tuple(padding.flatten())
