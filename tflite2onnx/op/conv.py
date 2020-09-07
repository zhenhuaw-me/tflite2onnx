import logging
import tflite

from tflite2onnx import tensor
from tflite2onnx.layout import Layout
from tflite2onnx.op.activation import handleFusedActivation
from tflite2onnx.op.operator import Operator
from tflite2onnx.op.padding import computePaddingSize

logger = logging.getLogger('tflite2onnx')


class Conv(Operator):
    def __init__(self, model, graph, index):
        super().__init__(model, graph, index)

        self.attrs['kernel_shape'] = []
        self.attrs['strides'] = []
        # ONNX: This attribute cannot be used simultaneously with `auto_pad` attribute.
        # re-initialize during self.parse(), as it needs the shape of input.
        # We prefer `auto_pad`, however ONNXRuntime doesn't support
        # `dilation` + `auto_pad`, such that we use `pads` to workaround it.
        self.attrs['pads'] = [0, 0, 0, 0]
        # XXX Not enabled as ONNXRuntime has limitation to infer pads for non-1 dilation
        # self.attrs['auto_pad'] = 'SAME_UPPER'  # See ComputePaddingHeightWidth() of TFLite
        self.attrs['dilations'] = []
        self.attrs['group'] = -1

        self.has_bias = True

        self.setInited()

    @property
    def type(self):
        if self.status.parsed and self.quantized:
            return 'QLinearConv'
        else:
            return 'Conv'

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
        wt = tensor.get(self.model, self.graph, wi, wlayout)
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
            bt = tensor.get(self.model, self.graph, bi, is_bias=True)
            bt.parse()
            bt.addConsumer(self)
            self.inputs.append(bt)

        # options
        op_opt = op.BuiltinOptions()
        option = tflite.DepthwiseConv2DOptions() if self.isDepthwise else tflite.Conv2DOptions()
        option.Init(op_opt.Bytes, op_opt.Pos)

        self.attrs['dilations'] = [option.DilationHFactor(), option.DilationWFactor()]
        self.attrs['group'] = it.shape[3] if self.isDepthwise else 1
        self.attrs['kernel_shape'] = wt.shape[1:3]
        self.attrs['strides'] = [option.StrideH(), option.StrideW()]
        # XXX Not enabled as ONNXRuntime has limitation to infer pads for non-1 dilation
        # self.attrs['auto_pad'] = PaddingMapping[option.Padding()]
        if self.isDepthwise:
            assert(option.DepthMultiplier() == 1)
        self.attrs['pads'] = computePaddingSize(option.Padding(), it.shape[1:3],
                                                self.attrs['kernel_shape'],
                                                self.attrs['strides'], self.attrs['dilations'])

        handleFusedActivation(self, option, ot)

        self.setParsed()

    def propagatableTensors(self):
        return list()

    @property
    def quantized(self):
        return False
        return tensor.isTFLiteQuantized(self.graph, self.tflite.Outputs(0))

    def transform(self):
        pass
