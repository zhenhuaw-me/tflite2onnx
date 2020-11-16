import logging
import tflite

from tflite2onnx.op.activation import handleFusedActivation
from tflite2onnx.op.common import Operator

logger = logging.getLogger('tflite2onnx')


class Concat(Operator):
    TypeMapping = {
        tflite.BuiltinOperator.CONCATENATION: 'Concat',
    }

    def __init__(self, TFactory, index):
        super().__init__(TFactory, index)

        self.attrs['axis'] = -1

        self.setInited()

    @property
    def type(self):
        return 'Concat'

    def parse(self):
        logger.debug("Parsing %s...", self.shorty)
        op = self.tflite
        opcode = self.model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
        assert(opcode in self.TypeMapping)

        assert(op.InputsLength() >= 1)
        assert(op.OutputsLength() == 1)

        for i in range(op.InputsLength()):
            self.parseInput(i)

        op_opt = op.BuiltinOptions()
        option = tflite.ConcatenationOptions()
        option.Init(op_opt.Bytes, op_opt.Pos)
        self.attrs['axis'] = option.Axis()

        self.parseOutput(0)

        handleFusedActivation(self, option, self.outputs[0])

        self.setParsed()

    def propagatableTensors(self):
        return self.inputs + self.outputs

    def transform(self):
        logger.debug("Transforming %s...", self.shorty)
        layout = self.outputs[0].layout
        if layout is not None:
            axis = self.attrs['axis']
            axis = axis if axis >= 0 else (axis + len(layout.perm))
            self.attrs['axis'] = layout.perm.index(axis)
