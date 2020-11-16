import logging
import tflite

from tflite2onnx.tensor import TensorFactory
from tflite2onnx.op.common import Operator

logger = logging.getLogger('tflite2onnx')


class Transpose(Operator):
    TypeMapping = {
        tflite.BuiltinOperator.TRANSPOSE: 'Transpose'
    }

    def __init__(self, TFactory, index):
        super().__init__(TFactory, index)

        self.attrs['perm'] = []

        self.setInited()

    @property
    def type(self):
        return 'Transpose'

    def parse(self):
        logger.debug("Parsing %s...", self.type)
        op = self.tflite
        opcode = self.model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
        assert(opcode in self.TypeMapping)

        assert(op.InputsLength() == 2)
        assert(op.OutputsLength() == 1)
        self.parseInput(0)
        self.parseOutput(0)

        ii = op.Inputs(1)
        self.attrs['perm'] = TensorFactory.getData(self.model, self.graph, ii, 'int32')

        self.setParsed()

    def propagatableTensors(self):
        return list()

    def transform(self):
        logger.warning("Transforming %s, doing nothing now...", self.type)
