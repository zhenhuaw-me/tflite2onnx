import numpy as np
import tflite
from onnx import helper

from .. import tensor
from .. import layout
from ..common import logger
from .op import Operator


OpTypeMapping = {
        tflite.BuiltinOperator.TRANSPOSE : 'Transpose',     # noqa: E203
}


class Transpose(Operator):
    def __init__(self, model, graph, index):
        super().__init__(model, graph, index)

        self.perm = []

        self.setInited()

    @property
    def type(self):
        return 'Transpose'

    @property
    def sensitive(self):
        return True

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
        self.perm = tensor.getData(self.model, self.graph, ii, np.int32)

        oi = op.Outputs(0)
        ot = tensor.get(self.model, self.graph, oi)
        ot.parse()
        ot.addProducer(self)
        self.outputs.append(ot)

        self.setParsed()

    def propagate(self):
        logger.debug("Propagating %s...", self.type)
        self.setPropagated()

    def convert(self):
        self.propagate()
        logger.debug("Converting %s...", self.type)

        self.inputs[0].convert()
        self.outputs[0].convert()

        inames = [t.name for t in self.inputs]
        onames = [t.name for t in self.outputs]
        logger.debug("Making ONNX...")
        self.onnx = helper.make_node(self.type, inames, onames, perm=self.perm)


def createTransposeHelper(input, output, upstream):
    logger.debug("Creating layout helper for <%s> -> <%s>", input.name, output.name)
    op = Transpose(input.model, input.graph, 0)
    op.index = -1
    op.tflite = None
    op.name = 'Layout Helper'
    op.inputs.append(input)
    op.outputs.append(output)
    if upstream:
        l = output.layout
        op.perm = layout.getPerm(l.target, l.source)
    else:
        l = input.layout
        op.perm = layout.getPerm(l.source, l.target)
    op.setParsed()

    input.addConsumer(op)
    output.addProducer(op)

    return op

