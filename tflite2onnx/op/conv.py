import logging
import tflite

from tflite2onnx.layout import Layout
from tflite2onnx.op.activation import handleFusedActivation
from tflite2onnx.op.common import Operator
from tflite2onnx.op.padding import computePaddingSize
from tflite2onnx.op.padding import PaddingMapping

logger = logging.getLogger('tflite2onnx')


class Conv(Operator):
    TypeMapping = {
        tflite.BuiltinOperator.CONV_2D: 'Conv',
        tflite.BuiltinOperator.DEPTHWISE_CONV_2D: 'Conv',
    }

    def __init__(self, TFactory, index):
        super().__init__(TFactory, index)

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

        self.setInited()

    @property
    def type(self):
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
        assert(opcode in self.TypeMapping)

        assert(op.InputsLength() == 3), "TFLite Conv always has bias"
        assert(op.OutputsLength() == 1)

        # input
        ilayout = Layout('NHWC', 'NCHW')
        it = self.parseInput(0, ilayout)

        # weight
        wlayout = Layout('CHWM', 'MCHW') if self.isDepthwise else Layout('OHWI', 'OIHW')
        wt = self.parseInput(1, wlayout)

        # bias
        self.parseInput(2, is_bias=True)

        # output
        olayout = Layout('NHWC', 'NCHW')
        ot = self.parseOutput(0, olayout)

        # options
        op_opt = op.BuiltinOptions()
        option = tflite.DepthwiseConv2DOptions() if self.isDepthwise else tflite.Conv2DOptions()
        option.Init(op_opt.Bytes, op_opt.Pos)

        self.attrs['dilations'] = [option.DilationHFactor(), option.DilationWFactor()]
        self.attrs['group'] = wt.shape[3] if self.isDepthwise else 1
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

    def transform(self):
        pass


class TransposeConv(Operator):
    TypeMapping = {
        tflite.BuiltinOperator.TRANSPOSE_CONV: 'ConvTranspose',
    }

    def __init__(self, TFactory, index):
        super().__init__(TFactory, index)

        self.attrs['dilations'] = []
        self.attrs['group'] = -1
        self.attrs['output_shape'] = []
        self.attrs['kernel_shape'] = []
        self.attrs['strides'] = []
        self.attrs['auto_pad'] = 'SAME_UPPER'  # See ComputePaddingHeightWidth() of TFLite

        self.setInited()

    @property
    def type(self):
        return 'ConvTranspose'


    def parse(self):
        logger.debug("Parsing %s...", self.type)
        op = self.tflite
        opcode = self.model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()

        assert(opcode is tflite.BuiltinOperator.TRANSPOSE_CONV)
        assert(op.InputsLength() == 3)
        assert(op.OutputsLength() == 1)

        # X
        ilayout = Layout('NHWC', 'NCHW')
        self.parseInput(2, ilayout)

        # weight
        wlayout = Layout('CHWM', 'MCHW') 
        W = self.parseInput(1, wlayout)

        # output
        olayout = Layout('NHWC', 'NCHW')
        O = self.parseOutput(0, olayout)
        os = O.shape
        os_olayout = olayout.transform(os)

        # options
        op_opt = op.BuiltinOptions()
        option = tflite.TransposeConvOptions()
        option.Init(op_opt.Bytes, op_opt.Pos)

        self.attrs['output_shape'] = os_olayout
        self.attrs['dilations'] = [1, 1]
        self.attrs['group'] = 1
        self.attrs['auto_pad'] = PaddingMapping[option.Padding()]
        self.attrs['kernel_shape'] = W.shape[1:3]  
        self.attrs['strides'] = [option.StrideH(), option.StrideW()]

        self.setParsed()

    def propagatableTensors(self):
        return list()

    def transform(self):
        pass
