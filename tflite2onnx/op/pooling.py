import logging
import tflite
from onnx import helper

from tflite2onnx import tensor
from tflite2onnx.op.operator import Operator
from tflite2onnx.op.padding import PaddingMapping
from tflite2onnx.op.activation import handleFusedActivation
from tflite2onnx.layout import Layout

logger = logging.getLogger('tflite2onnx')


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

        op_opt = op.BuiltinOptions()
        option = tflite.Pool2DOptions()
        option.Init(op_opt.Bytes, op_opt.Pos)
        self.auto_pad = PaddingMapping[option.Padding()]

        self.kshape = [option.FilterHeight(), option.FilterWidth()]
        self.strides = [option.StrideH(), option.StrideW()]

        oi = op.Outputs(0)
        olayout = Layout('NHWC', 'NCHW')
        ot = tensor.get(self.model, self.graph, oi, olayout)
        ot.parse()

        handleFusedActivation(self, option, ot)

        # ot.addProducer(self)
        # self.outputs.append(ot)

        self.setParsed()

    def convert(self):
        logger.debug("Converting %s...", self.type)

        self.inputs[0].convert()
        self.outputs[0].convert()

        inames = [t.name for t in self.inputs]
        onames = [t.name for t in self.outputs]
        logger.debug("Making ONNX...")
        self.onnx = helper.make_node(self.type, inames, onames, kernel_shape=self.kshape,
                                     strides=self.strides, auto_pad=self.auto_pad)
