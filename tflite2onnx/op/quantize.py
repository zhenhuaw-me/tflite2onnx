import logging
import tflite
from onnx import TensorProto

from tflite2onnx.op.common import Operator

logger = logging.getLogger('tflite2onnx')


class Quantize(Operator):
    TypeMapping = {
        tflite.BuiltinOperator.QUANTIZE: 'QuantizeLinear',
        tflite.BuiltinOperator.DEQUANTIZE: 'DequantizeLinear',
    }

    def __init__(self, TFactory, index):
        super().__init__(TFactory, index)

        # self.axis = 1

        self.setInited()

    @property
    def type(self):
        return 'QuantizeLinear' if self.isQuantize else 'DequantizeLinear'

    @property
    def isQuantize(self):
        if self.status.parsed:
            return self.inputs[0].dtype is TensorProto.FLOAT
        else:
            # to cover the case when isQuantize is called from logger
            # it may happen before the actual parsing begins
            opcode = self.model.OperatorCodes(self.tflite.OpcodeIndex()).BuiltinCode()
            return opcode is tflite.BuiltinOperator.QUANTIZE

    def parse(self):
        logger.debug("Parsing %s...", self.shorty)
        op = self.tflite
        opcode = self.model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
        assert(opcode in self.TypeMapping)

        assert(op.InputsLength() == 1)
        assert(op.OutputsLength() == 1)
        self.parseInput(0)
        self.parseOutput(0)

        self.setParsed()

    def propagatableTensors(self):
        return self.inputs + self.outputs

    def dequantize(self):
        if self.isQuantize:
            ft = self.inputs[0]
            qt = self.outputs[0]
        else:
            qt = self.inputs[0]
            ft = self.outputs[0]

        ft.dequantize()
        # assert(qt.quantized)

        st = self.TFactory.createQuantScale(qt)
        st.addConsumer(self)
        self.inputs.append(st)
        zpt = self.TFactory.createQuantZeroPoint(qt)
        zpt.addConsumer(self)
        self.inputs.append(zpt)

    def transform(self):
        pass

    def validate(self):
        quant_dtype = self.outputs[0].dtype if self.isQuantize else self.inputs[0].dtype
        if quant_dtype not in [TensorProto.UINT8, TensorProto.INT8, TensorProto.INT32]:
            raise ValueError("Unsupported quantization type due to ONNX operator semantic. "
                             "See https://github.com/jackwish/tflite2onnx/blob/master/docs/faq.md#fp16-quantization-model-doesnt-work")  # noqa: E501
