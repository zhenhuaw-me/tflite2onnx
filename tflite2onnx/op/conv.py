import tflite
from onnx import helper

from .. import tensor
from ..common import logger
from .op import Operator
from ..layout import Layout


class Conv2D(Operator):
    def __init__(self, model, graph, index):
        super().__init__(model, graph, index)

        self.auto_pad = 'SAME_UPPER'  # See ComputePaddingHeightWidth() of TFLite
        self.dilations = []
        self.group = -1
        self.kshape = []
        self.strides = []
        # pads #  This attribute cannot be used simultaneously with auto_pad attribute.

        self.setInited()

    @property
    def type(self):
        return 'Conv'

    @property
    def sensitive(self):
        return True

    def parse(self):
        logger.debug("Parsing %s...", self.type)
        op = self.tflite
        opcode = self.model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
        assert(opcode is tflite.BuiltinOperator.CONV_2D)

        assert(op.InputsLength() == 3)
        assert(op.OutputsLength() == 1)

        # input
        ii = op.Inputs(0)
        ilayout = Layout('NHWC', 'NCHW')
        it = tensor.get(self.model, self.graph, ii, ilayout)
        it.parse()
        it.addConsumer(self)
        self.inputs.append(it)

        # weight
        wi = op.Inputs(1)
        wlayout = Layout('OHWI', 'OIHW')
        wt = tensor.get(self.model, self.graph, wi, wlayout, True)
        wt.parse()
        wt.addConsumer(self)
        self.inputs.append(wt)

        # bias
        bi = op.Inputs(2)
        bt = tensor.get(self.model, self.graph, bi, None, True)
        bt.parse()
        bt.addConsumer(self)
        self.inputs.append(bt)

        # options
        op_opt = op.BuiltinOptions()
        option = tflite.Conv2DOptions()
        option.Init(op_opt.Bytes, op_opt.Pos)

        self.dilations = [option.DilationHFactor(), option.DilationWFactor()]
        self.group = 1
        self.kshape = wt.shape[1:3]
        self.strides = [option.StrideH(), option.StrideW()]
        # option.FusedActivationFunction()
        padding = option.Padding()
        assert(padding == tflite.Padding.SAME)  # TODO: enable VALID padding

        # output
        oi = op.Outputs(0)
        olayout = Layout('NHWC', 'NCHW')
        ot = tensor.get(self.model, self.graph, oi, olayout)
        ot.parse()
        ot.addProducer(self)
        self.outputs.append(ot)

        self.setParsed()

    def convert(self):
        self.propagate()
        logger.debug("Converting %s...", self.type)

        super()._convertTensors()

        inames = [t.name for t in self.inputs]
        onames = [t.name for t in self.outputs]
        logger.debug("Making ONNX...")
        self.onnx = helper.make_node(self.type, inames, onames, kernel_shape=self.kshape,
                                     strides=self.strides, auto_pad=self.auto_pad,
                                     dilations=self.dilations, group=self.group)
