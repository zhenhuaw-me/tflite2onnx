import logging
import tflite
from onnx import helper

from tflite2onnx import tensor
from tflite2onnx.layout import Layout
from tflite2onnx.op.activation import handleFusedActivation
from tflite2onnx.op.operator import Operator
from tflite2onnx.op.padding import PaddingMapping

logger = logging.getLogger('tflite2onnx')


OpTypeMapping = {
    tflite.BuiltinOperator.AVERAGE_POOL_2D: 'AveragePool',
    tflite.BuiltinOperator.MAX_POOL_2D: 'MaxPool',
}


class Pooling(Operator):
    def __init__(self, model, graph, index):
        super().__init__(model, graph, index)

        self.auto_pad = 'SAME_UPPER'  # See ComputePaddingHeightWidth() of TFLite
        # ceil_mod = 0
        self.kshape = []
        self.strides = []

        self.setInited()

    @property
    def type(self):
        if self.status.uninitialized:
            return 'Pooling'
        else:
            op = self.tflite
            opcode = self.model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
            assert(opcode in OpTypeMapping)
            return OpTypeMapping[opcode]

    @property
    def implicitLayout(self):
        return True

    @property
    def layoutPropagatable(self):
        return False

    def parse(self):
        logger.debug("Parsing %s...", self.type)
        op = self.tflite
        opcode = self.model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
        assert(opcode in OpTypeMapping)

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
        ot.addProducer(self)
        self.outputs.append(ot)

        handleFusedActivation(self, option, ot)

        self.setParsed()

    def transform(self):
        pass

    def convert(self):
        logger.debug("Converting %s...", self.type)

        self.inputs[0].convert()
        self.outputs[0].convert()

        inames = [t.name for t in self.inputs]
        onames = [t.name for t in self.outputs]
        logger.debug("Making ONNX...")
        self.onnx = helper.make_node(self.type, inames, onames, kernel_shape=self.kshape,
                                     strides=self.strides, auto_pad=self.auto_pad)
