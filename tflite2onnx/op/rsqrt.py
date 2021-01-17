import logging
import tflite
import numpy as np

from tflite2onnx import mapping
from tflite2onnx.op.common import Operator
from tflite2onnx.op.binary import PowerWrapper

logger = logging.getLogger('tflite2onnx')


class Rsqrt(Operator):
    # use square root as input operator and propagate output to power
    TypeMapping = {
        tflite.BuiltinOperator.RSQRT: 'Sqrt',
    }

    def __init__(self, TFactory, index):
        super().__init__(TFactory, index)
        self.setInited()

    @property
    def type(self):
        if self.status.uninitialized:
            return 'Sqrt'
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

        assert(op.InputsLength() == 1)
        assert(op.OutputsLength() == 1)

        self.parseInput(0)
        self.parseOutput(0)

        # invert square root result
        self.appendInvert()

        self.setParsed()

    def propagatableTensors(self):
        return self.inputs + self.outputs

    def transform(self):
        pass

    def appendInvert(self):
        invert = PowerWrapper(self.TFactory, -1)

        invert_name = 'TFLITE2ONNX_Invert_%s' % self.outputs[0].name
        invert_t = self.TFactory.getWithRef(self.outputs[0], invert_name, True)
        invert_t.setParsed()
        invert_t.addProducer(self)
        invert_t.addConsumer(invert)

        pow_t = 'TFLITE2ONNX_PowData_%s' % self.outputs[0].name
        pow_t = self.TFactory.getWithRef(self.outputs[0], pow_t, True)
        pow_dtype = mapping.DTYPE_ONNX2NAME[pow_t.dtype]
        pow_t.data = np.full(shape=pow_t.shape, fill_value=-1, dtype=pow_dtype)
        pow_t.setParsed()
        pow_t.addConsumer(invert)

        invert.inputs.append(invert_t)
        invert.inputs.append(pow_t)
        invert.outputs.append(self.outputs[0])
        self.replaceOutput(self.outputs[0], invert_t)

        invert.setParsed()
        self.post.append(invert)
