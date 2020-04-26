import tflite
from onnx import helper

from .. import tensor
from ..common import logger, Status
from .op import Operator


OpTypeMapping = {
        tflite.BuiltinOperator.ADD : 'Add',     # noqa: E203
}


class Binary(Operator):
    def __init__(self, model, graph, index):
        super().__init__(model, graph, index)
        logger.debug("Converting...")
        self.setInited()

    @property
    def type(self):
        if (self.status is Status.UNINITIALIZED):
            return 'Binary'
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

        assert(op.InputsLength() == 2)
        assert(op.OutputsLength() == 1)

        for i in range(op.InputsLength()):
            ii = op.Inputs(i)
            it = tensor.get(self.model, self.graph, ii)
            it.parse()
            self.inputs.append(it)

        oi = op.Outputs(0)
        ot = tensor.get(self.model, self.graph, oi)
        ot.parse()
        self.outputs.append(ot)

        self.setParsed()

    def buildGraph(self):
        logger.debug("Building graph in %s...", self.type)
        self.setGraphBuilt()

    def propagate(self):
        logger.debug("Propagating %s...", self.type)
        self.setPropagated()

    def convert(self):
        logger.debug("Converting %s...", self.type)
        self.buildGraph()
        self.propagate()

        for t in self.inputs:
            t.convert()
        self.outputs[0].convert()

        inames = [t.name for t in self.inputs]
        onames = [t.name for t in self.outputs]
        self.onnx = helper.make_node(self.type, inames, onames)
        self.setConverted()
