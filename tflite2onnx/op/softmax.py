import logging
import tflite

from tflite2onnx.op.common import Operator

logger = logging.getLogger('tflite2onnx')


class Softmax(Operator):
    TypeMapping = {
            tflite.BuiltinOperator.SOFTMAX: 'Softmax',
    }

    def __init__(self, TFactory, index):
        super().__init__(TFactory, index)

        self.attrs['axis'] = -1

        self.setInited()

    @property
    def type(self):
        return 'Softmax'

    def parse(self):
        logger.debug("Parsing %s...", self.type)
        op = self.tflite
        opcode = self.model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
        assert(opcode in self.TypeMapping)

        assert(op.InputsLength() == 1)
        assert(op.OutputsLength() == 1)
        self.parseInput(0)
        self.parseOutput(0)

        # TFLite Softmax ALWAYS softmax on `-1` axis, while ONNX on `1` by default.
        # And, we transform NHWC to NCHW for 4D tensor.
        # axis = 1 if len(to.shape) == 4 else -1
        # if len(to.shape) == 4:
        #     axis = 1
        # elif len(to.shape) == 2:
        #     axis = -1
        # else:
        #     axis = -1
        #     logger.warning("Softmax has input shape %s.", str(to.shape))
        # FIXME: axis is the dim index of 'C'.
        self.attrs['axis'] = -1

        self.setParsed()

    def propagatableTensors(self):
        return self.inputs + self.outputs

    def transform(self):
        pass
