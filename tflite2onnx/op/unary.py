import logging
import tflite

from tflite2onnx.op.common import Operator

logger = logging.getLogger('tflite2onnx')


OpTypeMapping = {
    tflite.BuiltinOperator.ABS: 'Abs',
}


class Unary(Operator):
    def __init__(self, model, graph, tregistry, index):
        super().__init__(model, graph, tregistry, index)
        self.setInited()

    @property
    def type(self):
        if self.status.uninitialized:
            return 'Unary'
        else:
            op = self.tflite
            opcode = self.model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
            assert(opcode in OpTypeMapping)
            return OpTypeMapping[opcode]

    def parse(self):
        logger.debug("Parsing %s...", self.type)

        op = self.tflite
        opcode = self.model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
        assert(opcode in OpTypeMapping)

        assert(op.InputsLength() == 1)
        assert(op.OutputsLength() == 1)
        self.parseInput(0)
        self.parseOutput(0)

        self.setParsed()

    def propagatableTensors(self):
        return self.inputs + self.outputs

    def transform(self):
        pass
