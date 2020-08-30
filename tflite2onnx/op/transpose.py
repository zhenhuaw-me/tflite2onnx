import logging
import tflite

from tflite2onnx import tensor
from tflite2onnx.op.operator import Operator

logger = logging.getLogger('tflite2onnx')


class Transpose(Operator):
    def __init__(self, model, graph, index):
        super().__init__(model, graph, index)

        self.attrs['perm'] = []

        self.setInited()

    @property
    def type(self):
        return 'Transpose'

    @property
    def layoutPropagatable(self):
        return False

    def parse(self):
        logger.debug("Parsing %s...", self.type)
        op = self.tflite
        opcode = self.model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
        assert(opcode is tflite.BuiltinOperator.TRANSPOSE)

        assert(op.InputsLength() == 2)
        assert(op.OutputsLength() == 1)

        ii = op.Inputs(0)
        it = tensor.get(self.model, self.graph, ii)
        it.parse()
        it.addConsumer(self)
        self.inputs.append(it)

        ii = op.Inputs(1)
        self.attrs['perm'] = tensor.getData(self.model, self.graph, ii, 'int32')

        oi = op.Outputs(0)
        ot = tensor.get(self.model, self.graph, oi)
        ot.parse()
        ot.addProducer(self)
        self.outputs.append(ot)

        self.setParsed()

    def transform(self):
        logger.warning("Transforming %s, doing nothing now...", self.type)
