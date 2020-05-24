import logging
import tflite
from onnx import helper

from tflite2onnx import tensor
from tflite2onnx.op.operator import Operator
from tflite2onnx.op.activation import createFusedActivation
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
        # pads #  This attribute cannot be used simultaneously with auto_pad attribute.

        self.setInited()

    @property
    def type(self):
        return 'Conv'

    @property
    def sensitive(self):
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
        wlayout = Layout('CHWM', 'MCHW') if self.isDepthwise else Layout('OHWI', 'OIHW')
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
        option = tflite.DepthwiseConv2DOptions() if self.isDepthwise else tflite.Conv2DOptions()
        option.Init(op_opt.Bytes, op_opt.Pos)

        self.dilations = [option.DilationHFactor(), option.DilationWFactor()]
        self.group = it.shape[3] if self.isDepthwise else 1
        self.kshape = wt.shape[1:3]
        self.strides = [option.StrideH(), option.StrideW()]
        padding = option.Padding()
        assert(padding == tflite.Padding.SAME)  # TODO: enable VALID padding
        if self.isDepthwise:
            assert(option.DepthMultiplier() == 1)

        # output
        oi = op.Outputs(0)
        olayout = Layout('NHWC', 'NCHW')
        ot = tensor.get(self.model, self.graph, oi, olayout)
        ot.parse()

        actf = option.FusedActivationFunction()
        if actf is tflite.ActivationFunctionType.RELU6:
            act = createFusedActivation(self.model, self.graph, actf, ot)
            it_act = act.inputs[0]
            it_act.addProducer(self)
            self.outputs.append(it_act)
            self.post.append(act)
        elif actf is tflite.ActivationFunctionType.NONE:
            ot.addProducer(self)
            self.outputs.append(ot)
        else:
            raise NotImplementedError("Unsupported fused ActivationFunctionType")

        self.setParsed()

    def convert(self):
        logger.debug("Converting %s...", self.type)

        super()._convertTensors()

        inames = [t.name for t in self.inputs]
        onames = [t.name for t in self.outputs]
        logger.debug("Making ONNX...")
        self.onnx = helper.make_node(self.type, inames, onames, kernel_shape=self.kshape,
                                     strides=self.strides, auto_pad=self.auto_pad,
                                     dilations=self.dilations, group=self.group)

    def __str__(self):
        attrs = 'K%s, S%s, D%s, G(%d)' % (self.kshape, self.strides, self.dilations, self.group)
        return super().__str__() + attrs
