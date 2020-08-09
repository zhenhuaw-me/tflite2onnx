import logging
import tflite
from onnx import helper

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
        return 'QuantizeLinear'

    @property
    def implicitLayout(self):
        return False

    def parse(self):
        logger.debug("Parsing %s...", self.type)
        op = self.tflite
        opcode = self.model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
        assert(opcode is tflite.BuiltinOperator.QUANTIZE)

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
        it = self.inputs[0]
        it.dequantize()
        ot = self.outputs[0]
        assert(ot.quantized)

        st = tensor.createQuantScale(ot)
        st.addConsumer(self)
        self.inputs.append(st)
        zpt = tensor.createQuantZeroPoint(ot)
        zpt.addConsumer(self)
        self.inputs.append(zpt)

    def transform(self):
        pass

    def convert(self):
        logger.debug("Converting %s...", self.type)

        self.inputs[0].convert()
        self.outputs[0].convert()

        inames = [t.name for t in self.inputs]
        onames = [t.name for t in self.outputs]
        self.onnx = helper.make_node(self.type, inames, onames)
        self.setConverted()


class Dequantize(Operator):
    def __init__(self, model, graph, index):
        super().__init__(model, graph, index)

        self.axis = 1

        self.setInited()

    @property
    def type(self):
        return 'DequantizeLinear'

    @property
    def implicitLayout(self):
        return False

    def parse(self):
        logger.debug("Parsing %s...", self.type)
        op = self.tflite
        opcode = self.model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
        assert(opcode is tflite.BuiltinOperator.DEQUANTIZE)

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
        it = self.inputs[0]
        assert(it.quantized)
        ot = self.outputs[0]
        ot.dequantize()

        st = tensor.createQuantScale(it)
        st.addConsumer(self)
        self.inputs.append(st)
        zpt = tensor.createQuantZeroPoint(it)
        zpt.addConsumer(self)
        self.inputs.append(zpt)

    def transform(self):
        pass

    def convert(self):
        logger.debug("Converting %s...", self.type)

        self.inputs[0].convert()
        self.outputs[0].convert()

        inames = [t.name for t in self.inputs]
        onames = [t.name for t in self.outputs]
        self.onnx = helper.make_node(self.type, inames, onames)
        self.setConverted()
