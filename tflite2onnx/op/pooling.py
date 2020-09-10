import logging
import tflite

from tflite2onnx.layout import Layout
from tflite2onnx.op.activation import handleFusedActivation
from tflite2onnx.op.common import Operator
from tflite2onnx.op.padding import PaddingMapping

logger = logging.getLogger('tflite2onnx')


class Pooling(Operator):
    TypeMapping = {
        tflite.BuiltinOperator.AVERAGE_POOL_2D: 'AveragePool',
        tflite.BuiltinOperator.MAX_POOL_2D: 'MaxPool',
    }

    def __init__(self, TFactory, index):
        super().__init__(TFactory, index)

        self.attrs['kernel_shape'] = []
        self.attrs['strides'] = []
        self.attrs['auto_pad'] = 'SAME_UPPER'  # See ComputePaddingHeightWidth() of TFLite
        # ceil_mod = 0

        self.setInited()

    @property
    def type(self):
        if self.status.uninitialized:
            return 'Pooling'
        else:
            op = self.tflite
            opcode = self.model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
            assert(opcode in self.TypeMapping)
            return self.TypeMapping[opcode]

    def parse(self):
        logger.debug("Parsing %s...", self.type)
        op = self.tflite
        opcode = self.model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
        assert(opcode in self.TypeMapping)

        assert(op.InputsLength() == 1)
        assert(op.OutputsLength() == 1)

        ilayout = Layout('NHWC', 'NCHW')
        self.parseInput(0, ilayout)

        op_opt = op.BuiltinOptions()
        option = tflite.Pool2DOptions()
        option.Init(op_opt.Bytes, op_opt.Pos)
        self.attrs['auto_pad'] = PaddingMapping[option.Padding()]
        self.attrs['kernel_shape'] = [option.FilterHeight(), option.FilterWidth()]
        self.attrs['strides'] = [option.StrideH(), option.StrideW()]

        olayout = Layout('NHWC', 'NCHW')
        ot = self.parseOutput(0, olayout)

        handleFusedActivation(self, option, ot)

        self.setParsed()

    def propagatableTensors(self):
        return list()

    def transform(self):
        pass
