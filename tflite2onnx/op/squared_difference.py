import logging
import tflite
import numpy as np

from tflite2onnx import mapping
from tflite2onnx.op.common import Operator
from tflite2onnx.op.binary import Binary

logger = logging.getLogger('tflite2onnx')


# wrapper is used here to override Binary.type property
class PowerWrapper(Binary):
    @property
    def type(self):
        return 'Pow'


class SquaredDifference(Operator):
    # use subtraction as input operator and propagate output to power
    TypeMapping = {
        tflite.BuiltinOperator.SQUARED_DIFFERENCE: 'Sub',
    }

    def __init__(self, TFactory, index):
        super().__init__(TFactory, index)
        self.setInited()

    @property
    def type(self):
        if self.status.uninitialized:
            return 'Sub'
        else:
            op = self.tflite
            opcode = self.model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
            assert(opcode in self.TypeMapping)
            return self.TypeMapping[opcode]

    def parse(self):
        logger.debug("Parsing %s...", self.type)

        op = self.tflite
        opcode = self.model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
        assert(opcode in self.TypeMapping)

        assert(op.InputsLength() == 2)
        assert(op.OutputsLength() == 1)

        self.parseInput(0)
        self.parseInput(1)
        self.parseOutput(0)

        assert(len(self.inputs[0].shape) == len(self.inputs[1].shape))

        # apply square to the subtraction result
        self.appendSquare()

        self.setParsed()

    def propagatableTensors(self):
        return self.inputs + self.outputs

    def transform(self):
        pass

    def appendSquare(self):
        square = PowerWrapper(self.TFactory, -1)

        square_name = 'TFLITE2ONNX_Square_%s' % self.outputs[0].name
        square_t = self.TFactory.getWithRef(self.outputs[0], square_name, True)
        square_t.setParsed()
        square_t.addProducer(self)
        square_t.addConsumer(square)

        pow_t = 'TFLITE2ONNX_PowData_%s' % self.outputs[0].name
        pow_t = self.TFactory.getWithRef(self.outputs[0], pow_t, True)
        pow_dtype = mapping.DTYPE_ONNX2NAME[pow_t.dtype]
        pow_t.data = np.full(shape=pow_t.shape, fill_value=2, dtype=pow_dtype)
        pow_t.setParsed()
        pow_t.addConsumer(square)

        square.inputs.append(square_t)
        square.inputs.append(pow_t)
        square.outputs.append(self.outputs[0])
        self.replaceOutput(self.outputs[0], square_t)

        square.setParsed()
        self.post.append(square)
