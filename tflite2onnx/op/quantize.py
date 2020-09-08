import logging
import tflite
from onnx import TensorProto

from tflite2onnx import tensor
from tflite2onnx.op.operator import Operator

logger = logging.getLogger('tflite2onnx')


class Quantize(Operator):
    def __init__(self, model, graph, index):
        super().__init__(model, graph, index)

        # self.axis = 1

        self.setInited()

    @property
    def type(self):
        return 'QuantizeLinear' if self.isQuantize else 'DequantizeLinear'

    @property
    def isQuantize(self):
        assert(self.status.parsed)
        return self.inputs[0].dtype is TensorProto.FLOAT

    def parse(self):
        logger.debug("Parsing %s...", self.shorty)
        op = self.tflite
        opcode = self.model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
        assert(opcode is tflite.BuiltinOperator.QUANTIZE or tflite.BuiltinOperator.DEQUANTIZE)

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

        st = tensor.createQuantScale(qt)
        st.addConsumer(self)
        self.inputs.append(st)
        zpt = tensor.createQuantZeroPoint(qt)
        zpt.addConsumer(self)
        self.inputs.append(zpt)

    def transform(self):
        pass
