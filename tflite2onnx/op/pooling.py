import tflite
from onnx import helper

from .. import tensor
from ..common import logger
from .op import Operator
from .transpose import TransposeHelper


OpTypeMapping = {
        tflite.BuiltinOperator.AVERAGE_POOL_2D : 'AveragePool',     # noqa: E203
}


class AveragePool(Operator):
    def __init__(self, model, graph, op):
        Operator.__init__(self)

    def convert(self, model, graph, op):
        logger.debug("Converting...")
        self.tflite = op
        opcode = model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
        assert(opcode in OpTypeMapping)
        self.type = OpTypeMapping[opcode]

        assert(op.InputsLength() == 1)
        assert(op.OutputsLength() == 1)

        ti = op.Inputs(0)
        tensor.convert(model, graph, ti)

        # NHWC -> Transpose -> NCHW
        inputTranspose = TransposeHelper(model, graph, op, 'NHWC', 'NCHW', iIndex=ti)

        # use output of inputTranspose as op input
        self.inputs.append(inputTranspose.outputs[0])

        op_opt = op.BuiltinOptions()
        option = tflite.Pool2DOptions()
        option.Init(op_opt.Bytes, op_opt.Pos)

        auto_pad = 'SAME_UPPER'  # See ComputePaddingHeightWidth() of TFLite
        # ceil_mod = 0
        kshape = [option.FilterHeight(), option.FilterWidth()]
        strides = [option.StrideH(), option.StrideW()]

        ti = op.Outputs(0)
        tensor.convert(model, graph, ti)

        # NCHW -> Transpose -> NHWC
        outputTranspose = TransposeHelper(model, graph, op, 'NCHW', 'NHWC', oIndex=ti)

        # use input of outputTranspose as op output
        self.outputs.append(outputTranspose.inputs[0])

        inames = [t.name for t in self.inputs]
        onames = [t.name for t in self.outputs]
        logger.debug("Making ONNX...")
        self.onnx = helper.make_node(self.type, inames, onames, kernel_shape=kshape,
                                     strides=strides, auto_pad=auto_pad)

        return [inputTranspose, self, outputTranspose]
