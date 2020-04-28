import tflite
from onnx import helper

from .. import tensor
from ..common import logger
from .op import Operator
from .transpose import TransposeHelper
from ..layout import Layout


class AveragePool(Operator):
    def __init__(self, model, graph, index):
        super().__init__(model, graph, index)

        self.auto_pad = 'SAME_UPPER'  # See ComputePaddingHeightWidth() of TFLite
        # ceil_mod = 0
        self.kshape = []
        self.strides = []

        self.setInited()

    @property
    def type(self):
        return 'AveragePool'

    @property
    def sensitive(self):
        return True

    def parse(self):
        logger.debug("Parsing %s...", self.type)
        op = self.tflite
        opcode = self.model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
        assert(opcode is tflite.BuiltinOperator.AVERAGE_POOL_2D)

        assert(op.InputsLength() == 1)
        assert(op.OutputsLength() == 1)

        ii = op.Inputs(0)
        ilayout = Layout('NHWC', 'NCHW')
        it = tensor.get(self.model, self.graph, ii, ilayout)
        it.parse()
        it.addConsumer(self)
        self.inputs.append(it)

        # # NHWC -> Transpose -> NCHW
        # inputTranspose = TransposeHelper(self.model, self.graph, self.index, 'NHWC', 'NCHW', iIndex=ti)

        # # use output of inputTranspose as op input
        # self.inputs.append(inputTranspose.outputs[0])

        op_opt = op.BuiltinOptions()
        option = tflite.Pool2DOptions()
        option.Init(op_opt.Bytes, op_opt.Pos)

        self.kshape = [option.FilterHeight(), option.FilterWidth()]
        self.strides = [option.StrideH(), option.StrideW()]

        oi = op.Outputs(0)
        olayout = Layout('NCHW', 'NHWC')
        ot = tensor.get(self.model, self.graph, oi, olayout)
        ot.parse()
        ot.addProducer(self)
        self.outputs.append(ot)

        # # NCHW -> Transpose -> NHWC
        # outputTranspose = TransposeHelper(self.model, self.graph, self.index, 'NCHW', 'NHWC', oIndex=ti)

        # # use input of outputTranspose as op output
        # self.outputs.append(outputTranspose.inputs[0])

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
        self.onnx = helper.make_node(self.type, inames, onames, kernel_shape=self.kshape,
                                     strides=self.strides, auto_pad=self.auto_pad)

        # return [inputTranspose, self, outputTranspose]
