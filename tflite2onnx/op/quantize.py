import logging
import tflite
from onnx import helper, TensorProto

from tflite2onnx import tensor
from tflite2onnx.op.operator import Operator

logger = logging.getLogger('tflite2onnx')


class Quantize(Operator):
    def __init__(self, model, graph, index):
        super().__init__(model, graph, index)

        self.axis = 1

        self.setInited()

    @property
    def type(self):
        return 'QuantizeLinear' if self.isQuantize else 'DequantizeLinear'

    @property
    def isQuantize(self):
        assert(self.status.parsed)
        return self.inputs[0].dtype is TensorProto.FLOAT

    @property
    def layoutPropagatable(self):
        return True

    def parse(self):
        logger.debug("Parsing %s...", self.str)
        op = self.tflite
        opcode = self.model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
        assert(opcode is tflite.BuiltinOperator.QUANTIZE or tflite.BuiltinOperator.DEQUANTIZE)

        assert(op.InputsLength() == 1)
        assert(op.OutputsLength() == 1)

        ii = op.Inputs(0)
        it = tensor.get(self.model, self.graph, ii)
        it.parse()
        it.addConsumer(self)
        self.inputs.append(it)

        oi = op.Outputs(0)
        ot = tensor.get(self.model, self.graph, oi)
        ot.parse()
        ot.addProducer(self)
        self.outputs.append(ot)

        self.setParsed()

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

    def convert(self):
        logger.debug("Converting %s...", self.str)

        self._convertTensors()

        inames = [t.name for t in self.inputs]
        onames = [t.name for t in self.outputs]
        self.onnx = helper.make_node(self.type, inames, onames)
        self.setConverted()
