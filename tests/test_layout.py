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


def test_fake_broadcast():
    from tflite2onnx.op.binary import fakeBroadcast
    def check_fakeBroadcast(a0, b0, a1, b1):
        a, b = fakeBroadcast(a0, b0)
        assert(a == a1)
        assert(b == b1)

    # cases from: https://github.com/onnx/onnx/blob/master/docs/Broadcasting.md
    check_fakeBroadcast([2, 3, 4, 5], list(), [2, 3, 4, 5], [1, 1, 1, 1])
    check_fakeBroadcast([2, 3, 4, 5], [5,], [2, 3, 4, 5], [1, 1, 1, 5])
    check_fakeBroadcast([4, 5], [2, 3, 4, 5,], [1, 1, 4, 5], [2, 3, 4, 5])
    check_fakeBroadcast([1, 4, 5], [2, 3, 1, 1,], [1, 1, 4, 5], [2, 3, 1, 1])
    check_fakeBroadcast([3, 4, 5], [2, 1, 1, 1], [1, 3, 4, 5], [2, 1, 1, 1])



if __name__ == '__main__':
    test_transform()
    test_getPerm()
    test_fake_broadcast()
