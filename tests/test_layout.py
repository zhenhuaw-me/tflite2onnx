import logging
import shrub

shrub.util.formatLogging(logging.DEBUG)


def test_transform():
    from tflite2onnx.layout import transform
    assert(transform([1, 2, 6, 8], 'NCHW', 'NHWC') == [1, 6, 8, 2])
    assert(transform([1, 2, 6, 8], 'NHWC', 'NCHW') == [1, 8, 2, 6])


def test_getPerm():
    from tflite2onnx.layout import getPerm
    assert(getPerm('01', '01') == [0, 1])
    assert(getPerm('01', '10') == [1, 0])
    assert(getPerm('0123', '0123') == [0, 1, 2, 3])
    assert(getPerm('0123', '0312') == [0, 3, 1, 2])
    assert(getPerm('0123', '3021') == [3, 0, 2, 1])


def test_align_dimension():
    from tflite2onnx.op.binary import alignDimension
    # cases from: https://github.com/onnx/onnx/blob/master/docs/Broadcasting.md
    assert(alignDimension([2, 3, 4, 5], list()) == (False, [1, 1, 1, 1]))
    assert(alignDimension([2, 3, 4, 5], [5, ]) == (False, [1, 1, 1, 5]))
    assert(alignDimension([4, 5], [2, 3, 4, 5]) == (True, [1, 1, 4, 5]))
    assert(alignDimension([1, 4, 5], [2, 3, 1, 1]) == (True, [1, 1, 4, 5]))
    assert(alignDimension([3, 4, 5], [2, 1, 1, 1]) == (True, [1, 3, 4, 5]))


if __name__ == '__main__':
    test_transform()
    test_getPerm()
    test_align_dimension()
