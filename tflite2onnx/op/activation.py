import logging
import tflite
from onnx import helper

from tflite2onnx import tensor
from tflite2onnx.op.operator import Operator

logger = logging.getLogger('tflite2onnx')


OpTypeMapping = {
    tflite.BuiltinOperator.RELU: 'Relu',
    tflite.BuiltinOperator.RELU6: 'Clip',
}


class ReLU(Operator):
    def __init__(self, model, graph, index):
        super().__init__(model, graph, index)
        self.setInited()

    @property
    def type(self):
        if self.status.uninitialized:
            return 'Relu-all'
        else:
            op = self.tflite
            opcode = self.model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
            assert(opcode in OpTypeMapping)
            return OpTypeMapping[opcode]

    @property
    def sensitive(self):
        return False

    def parse(self):
        logger.debug("Parsing %s...", self.type)

        op = self.tflite
        opcode = self.model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
        assert(opcode in OpTypeMapping)

        assert(op.InputsLength() == 1)
        assert(op.OutputsLength() == 1)

        ii = op.Inputs(0)
        it = tensor.get(self.model, self.graph, ii)
        it.parse()
        it.addConsumer(self)
        self.inputs.append(it)

        if opcode == tflite.BuiltinOperator.RELU6:
            tmin = tensor.createScalar(it, 0)
            tmin.addConsumer(self)
            self.inputs.append(tmin)
            tmax = tensor.createScalar(it, 6)
            tmax.addConsumer(self)
            self.inputs.append(tmax)

        oi = op.Outputs(0)
        ot = tensor.get(self.model, self.graph, oi)
        ot.parse()
        ot.addProducer(self)
        self.outputs.append(ot)

        self.setParsed()

    def convert(self):
        logger.debug("Converting %s...", self.type)

        for t in self.inputs:
            t.convert()
        self.outputs[0].convert()

        inames = [t.name for t in self.inputs]
        onames = [t.name for t in self.outputs]
        self.onnx = helper.make_node(self.type, inames, onames)
        self.setConverted()
