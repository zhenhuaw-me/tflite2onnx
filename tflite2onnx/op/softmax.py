import tflite
from onnx import helper

from ..common import logger
from ..tensor import create_tensor
from .op import Operator


OpTypeMapping = {
        tflite.BuiltinOperator.SOFTMAX : 'Softmax',     # noqa: E203
}


class Softmax(Operator):
    def __init__(self, model, graph, op):
        Operator.__init__(self)
        logger.debug("Converting...")
        self.tflite = op
        opcode = model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
        assert(opcode in OpTypeMapping)
        self.type = OpTypeMapping[opcode]

        assert(op.InputsLength() == 1)
        assert(op.OutputsLength() == 1)

        ti = op.Inputs(0)
        to = create_tensor(model, graph, ti, False)
        self.inputs.append(to)

        # TFLite Softmax ALWAYS softmax on `-1` axis, while ONNX on `1` by default.
        # And, we transform NHWC to NCHW for 4D tensor.
        # axis = 1 if len(to.dims) == 4 else -1
        # if len(to.dims) == 4:
        #     axis = 1
        # elif len(to.dims) == 2:
        #     axis = -1
        # else:
        #     axis = -1
        #     logger.warning("Softmax has input shape %s.", str(to.dims))
        axis = -1

        ti = op.Outputs(0)
        to = create_tensor(model, graph, ti, False)
        self.outputs.append(to)

        inames = [t.name for t in self.inputs]
        onames = [t.name for t in self.outputs]
        logger.debug("Making ONNX...")
        self.onnx = helper.make_node(self.type, inames, onames, axis=axis)
