import logging
import numpy as np
import tflite

from tflite2onnx.op.common import Operator

logger = logging.getLogger('tflite2onnx')

PaddingMapping = {
    tflite.Padding.SAME: 'SAME_UPPER',
    tflite.Padding.VALID: 'VALID',
}


class Padding(Operator):
    TypeMapping = {
        tflite.BuiltinOperator.PAD: 'Pad',
        tflite.BuiltinOperator.MIRROR_PAD: 'Pad',
    }

    def __init__(self, TFactory, index):
        super().__init__(TFactory, index)

        self.attrs['mode'] = 'constant'

        self.setInited()

    @property
    def type(self):
        if self.status.uninitialized:
            return 'Pad'
        else:
            opcode = self.model.OperatorCodes(self.tflite.OpcodeIndex()).BuiltinCode()
            assert(opcode in self.TypeMapping)
            return self.TypeMapping[opcode]

    def parse(self):
        logger.debug("Parsing %s...", self.shorty)
        op = self.tflite
        opcode = self.model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
        assert(opcode in self.TypeMapping)

        if opcode is tflite.BuiltinOperator.MIRROR_PAD:
            self.attrs['mode'] = 'reflect'
        else:
            self.attrs['mode'] = 'constant'

        assert(op.InputsLength() == 2)
        assert(op.OutputsLength() == 1)

        it = self.parseInput(0)

        pt = self.parseInput(1)
        assert(len(pt.shape) == 2)
        assert(pt.shape[0] == len(it.shape))
        assert(pt.shape[1] == 2)
        assert(pt.isInitializer)
        # bridge semantic gap
        pt.asDtype('int64')

        self.parseOutput(0)

        self.setParsed()

    def propagatableTensors(self):
        return [self.inputs[0], self.outputs[0]]

    def transform(self):
        # Padding.transform() handls TFLite/ONNX semantic gap in addition to layout gap
        # TensorFlow (Lite) pads is `[n, 2]` where `[i, 0]` is _begin_ and `[i, 1]` is _end_
        # ONNX pads is `[n * 2]` sequenced as `[x1_begin, x2_begin,...,x1_end, x2_end,...]`
        layout = self.inputs[0].layout
        pt = self.inputs[1]
        pads = pt.data
        pads = np.reshape(pads, pt.shape)
        if layout is None:
            pads = np.transpose(pads)
        else:
            pads_begin = pads[:, 0]
            pads_end = pads[:, 1]
            pads_begin = layout.transform(pads_begin)
            pads_end = layout.transform(pads_end)
            pads = np.array([pads_begin, pads_end])
        pt.data = pads.flatten()
        pt.shape = [np.prod(pt.shape), ]


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
    oshape = oshape.astype('int')

    # infer the padding
    total_padding = np.add(np.multiply(np.subtract(oshape, ones), stride),
                           np.subtract(effective_filter_size, input_size))
    total_padding = np.maximum(total_padding, np.zeros_like(input_size))
    total_padding = total_padding.astype('int')

    # ONNX semantic
    pre_padding = total_padding // 2
    post_padding = np.subtract(total_padding, pre_padding)
    padding = np.concatenate((pre_padding, post_padding))

    return padding.flatten()
