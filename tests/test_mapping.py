from tflite2onnx import mapping


def test_mapping():
    assert(len(mapping.DTYPE_NAME2ONNX) == 8)
    assert(len(mapping.DTYPE_NAME2TFLITE) == 8)
    assert(len(mapping.DTYPE_ONNX2NAME) == 8)
    assert(len(mapping.DTYPE_ONNX2TFLITE) == 8)
    assert(len(mapping.DTYPE_TFLITE2NAME) == 8)
    assert(len(mapping.DTYPE_TFLITE2ONNX) == 8)


if __name__ == '__main__':
    test_mapping()
