import logging
import tflite
from onnx import helper

from tflite2onnx import tensor
from tflite2onnx.op.operator import Operator
from tflite2onnx.op.padding import PaddingMapping, computePaddingSize
from tflite2onnx.op.activation import handleFusedActivation
from tflite2onnx.layout import Layout

logger = logging.getLogger('tflite2onnx')


class Conv(Operator):
    def __init__(self, model, graph, index):
        super().__init__(model, graph, index)

        self.auto_pad = 'SAME_UPPER'  # See ComputePaddingHeightWidth() of TFLite
        self.dilations = []
        self.group = -1
        self.kshape = []
        self.strides = []
        self.has_bias = True

        # ONNX: This attribute cannot be used simultaneously with `auto_pad` attribute.
        # re-initialize during self.parse(), as it needs the shape of input.
        # We prefer `auto_pad`, however ONNXRuntime doesn't support
        # `dilation` + `auto_pad`, such that we use `pads` to workaround it.
        self.pads = [0, 0, 0, 0]

        self.setInited()

    @property
    def type(self):
        if self.status.parsed and self.quantized:
            return 'QLinearConv'
        else:
            return 'Conv'

    @property
    def implicitLayout(self):
        return True

    @property
    def isDepthwise(self):
        op = self.tflite
        opcode = self.model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
        return (opcode is tflite.BuiltinOperator.DEPTHWISE_CONV_2D)

    def parse(self):
        logger.debug("Parsing %s...", self.type)
        op = self.tflite
        opcode = self.model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
        assert(opcode is tflite.BuiltinOperator.CONV_2D or tflite.BuiltinOperator.DEPTHWISE_CONV_2D)

        self.has_bias = op.InputsLength() == 3
        assert(self.has_bias), "TFLite Conv always has bias"
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
        wlayout = Layout('CHWM', 'MCHW') if self.isDepthwise else Layout('OHWI', 'OIHW')
        wt = tensor.get(self.model, self.graph, wi, wlayout, True)
        wt.parse()
        wt.addConsumer(self)
        self.inputs.append(wt)

        # output
        oi = op.Outputs(0)
        olayout = Layout('NHWC', 'NCHW')
        ot = tensor.get(self.model, self.graph, oi, olayout)
        ot.parse()
        ot.addProducer(self)
        self.outputs.append(ot)

        # bias
        if self.has_bias:
            bi = op.Inputs(2)
            bt = tensor.get(self.model, self.graph, bi,
                            is_initializer=True, is_bias=True)
            bt.parse()
            bt.addConsumer(self)
            self.inputs.append(bt)

        # options
        op_opt = op.BuiltinOptions()
        option = tflite.DepthwiseConv2DOptions() if self.isDepthwise else tflite.Conv2DOptions()
        option.Init(op_opt.Bytes, op_opt.Pos)

        self.dilations = [option.DilationHFactor(), option.DilationWFactor()]
        self.group = it.shape[3] if self.isDepthwise else 1
        self.kshape = wt.shape[1:3]
        self.strides = [option.StrideH(), option.StrideW()]
        self.auto_pad = PaddingMapping[option.Padding()]
        if self.isDepthwise:
            assert(option.DepthMultiplier() == 1)
        self.pads = computePaddingSize(option.Padding(), it.shape[1:3], self.kshape,
                                       self.strides, self.dilations)

        handleFusedActivation(self, option, ot)

        self.setParsed()

    @property
    def quantized(self):
        return False
        return tensor.isTFLiteQuantized(self.graph, self.tflite.Outputs(0))

    def transform(self):
        pass

    def convert(self):
        logger.debug("Converting %s...", self.type)

        super()._convertTensors()

        inames = [t.name for t in self.inputs]
        onames = [t.name for t in self.outputs]
        logger.debug("Making ONNX...")
        self.onnx = helper.make_node(self.type, inames, onames, kernel_shape=self.kshape,
                                     strides=self.strides, pads=self.pads,
                                     # strides=self.strides, auto_pad=self.auto_pad,
                                     dilations=self.dilations, group=self.group)

    def __str__(self):
        attrs = 'K%s, S%s, D%s, G(%d)' % (self.kshape, self.strides, self.dilations, self.group)
        return super().__str__() + attrs
