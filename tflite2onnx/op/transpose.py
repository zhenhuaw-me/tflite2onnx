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
        logger.debug("Converting...")
        op = self.tflite
        opcode = model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
        assert(opcode in OpTypeMapping)
        self.type = OpTypeMapping[opcode]

        assert(op.InputsLength() == 2)
        assert(op.OutputsLength() == 1)

        ti = op.Inputs(0)
        to = tensor.convert(model, graph, ti)
        self.inputs.append(to)

        ti = op.Inputs(1)
        perm = tensor.getData(model, graph, ti, np.int32)

        ti = op.Outputs(0)
        to = tensor.convert(model, graph, ti)
        self.outputs.append(to)

        inames = [t.name for t in self.inputs]
        onames = [t.name for t in self.outputs]
        logger.debug("Making ONNX...")
        self.onnx = helper.make_node(self.type, inames, onames, perm=perm)


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
