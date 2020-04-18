import numpy as np
import tflite
from onnx import helper

from ..common import logger
from ..tensor import create_tensor
from .op import Operator


OpTypeMapping = {
        tflite.BuiltinOperator.TRANSPOSE : 'Transpose',     # noqa: E203
}


class Transpose(Operator):
    def __init__(self, model, graph, op):
        Operator.__init__(self)
        logger.debug("Converting...")
        self.tflite = op
        opcode = model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
        assert(opcode in OpTypeMapping)
        self.type = OpTypeMapping[opcode]

        assert(op.InputsLength() == 2)
        assert(op.OutputsLength() == 1)

        ti = op.Inputs(0)
        to = create_tensor(model, graph, ti)
        self.inputs.append(to)

        ti = op.Inputs(1)
        tperm = graph.Tensors(ti)
        b = model.Buffers(tperm.Buffer())
        perm = np.frombuffer(b.DataAsNumpy(), dtype=np.int32)

        ti = op.Outputs(0)
        to = create_tensor(model, graph, ti)
        self.outputs.append(to)

        inames = [t.name for t in self.inputs]
        onames = [t.name for t in self.outputs]
        logger.debug("Making ONNX...")
        self.onnx = helper.make_node(self.type, inames, onames, perm=perm)
