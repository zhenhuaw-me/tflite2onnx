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
        self.inputs.append(it)

        ii = op.Inputs(1)
        self.perm = tensor.getData(self.model, self.graph, ii, np.int32)

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

        self.inputs[0].convert()
        self.outputs[0].convert()

        inames = [t.name for t in self.inputs]
        onames = [t.name for t in self.outputs]
        logger.debug("Making ONNX...")
        self.onnx = helper.make_node(self.type, inames, onames, perm=self.perm)


class TransposeHelper(Operator):
    def __init__(self, model, graph, index, ilayout, olayout, iIndex=None, oIndex=None):
        super().__init__(model, graph, index)
        logger.debug("Converting...")
        op = graph.Operators(index)
        self.tflite = op  # the tflite operator that Transpose helps for.
        assert((iIndex is None) != (oIndex is None)), "One of this IO needs to be empty"
        opcode = tflite.BuiltinOperator.TRANSPOSE
        assert(opcode in OpTypeMapping)
        self.type = OpTypeMapping[opcode]

        if iIndex is None:
            iTensor = tensor.createTransposeTensor(model, graph, oIndex, ilayout, olayout)
        else:
            iTensor = tensor.convert(model, graph, iIndex)
        self.inputs.append(iTensor)

        if oIndex is None:
            oTensor = tensor.createTransposeTensor(model, graph, iIndex, ilayout, olayout)
        else:
            oTensor = tensor.convert(model, graph, oIndex)
        self.outputs.append(oTensor)

        perm = layout.getPerm(ilayout, olayout)

        inames = [t.name for t in self.inputs]
        onames = [t.name for t in self.outputs]
        logger.debug("Making ONNX...")
        self.onnx = helper.make_node(self.type, inames, onames, perm=perm)
