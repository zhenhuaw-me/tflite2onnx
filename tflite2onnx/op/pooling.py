import tflite
from onnx import helper

from ..common import logger
from ..tensor import create_tensor
from .op import Operator


OpTypeMapping = {
        tflite.BuiltinOperator.AVERAGE_POOL_2D : 'AveragePool',     # noqa: E203
}


class AveragePool(Operator):
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
        to = create_tensor(model, graph, ti)
        self.inputs.append(to)

        op_opt = op.BuiltinOptions()
        option = tflite.Pool2DOptions()
        option.Init(op_opt.Bytes, op_opt.Pos)

        auto_pad = 'SAME_UPPER' # See ComputePaddingHeightWidth() of TFLite
        ceil_mod = 0
        kshape = [option.FilterHeight(), option.FilterWidth()]
        strides = [option.StrideH(), option.StrideW()]

        ti = op.Outputs(0)
        to = create_tensor(model, graph, ti)
        self.outputs.append(to)

        inames = [t.name for t in self.inputs]
        onames = [t.name for t in self.outputs]
        logger.debug("Making ONNX...")
        self.onnx = helper.make_node(self.type, inames, onames, kernel_shape=kshape,
                                     strides=strides, auto_pad=auto_pad)
