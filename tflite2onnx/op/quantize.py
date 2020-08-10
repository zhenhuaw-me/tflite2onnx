import copy
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
        return (self.status.parsed and
                (self.outputs[0].dtype is TensorProto.UINT8))

    @property
    def implicitLayout(self):
        return False

    def parse(self):
        logger.debug("Parsing %s...", self.type)
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
        assert(qt.quantized)

        st = tensor.createQuantScale(qt)
        st.addConsumer(self)
        self.inputs.append(st)
        zpt = tensor.createQuantZeroPoint(qt)
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


def createQuantize(ftensor, post_op):
    logger.debug("Creating Quantize node for %s", str(post_op))
    name = 'TFLITE2ONNX_' + ftensor.name + '_' + 'quantized'
    assert(name not in tensor.registery)

    # create tensor
    qtensor = tensor.Tensor(ftensor.model, ftensor.graph, -1)
    qtensor.name = name
    qtensor.dtype = TensorProto.UINT8
    qtensor.layout = copy.deepcopy(ftensor.layout)
    qtensor.shape = copy.deepcopy(ftensor.shape)
    qtensor.scale = copy.deepcopy(ftensor.scale)
    qtensor.zero_point = copy.deepcopy(ftensor.zero_point)
    qtensor.setParsed()
    tensor.registery[qtensor.name] = qtensor

    # create Quantize op
    op = Quantize(post_op.model, post_op.graph, -1)
    op.name = name
    op.inputs.append(ftensor)
    op.outputs.append(qtensor)
    op.setParsed()

    ftensor.addConsumer(op)
    ftensor.removeConsumer(post_op)
    qtensor.addProducer(op)
    qtensor.addConsumer(post_op)
    post_op.replaceInput(ftensor, qtensor)

    return (qtensor, op)


def createDequantize(qtensor, pre_op):
    logger.debug("Creating Dequantize node for %s", str(pre_op))
    name = 'TFLITE2ONNX_' + qtensor.name + '_' + 'dequantized'
    assert(name not in tensor.registery)

    # create tensor
    ftensor = tensor.Tensor(qtensor.model, qtensor.graph, -1)
    ftensor.name = name
    ftensor.dtype = TensorProto.FLOAT
    ftensor.layout = copy.deepcopy(qtensor.layout)
    ftensor.shape = copy.deepcopy(qtensor.shape)
    ftensor.scale = copy.deepcopy(qtensor.scale)
    ftensor.zero_point = copy.deepcopy(qtensor.zero_point)
    ftensor.setParsed()
    tensor.registery[qtensor.name] = ftensor

    # create Dequantize op
    op = Quantize(pre_op.model, pre_op.graph, -1)
    op.name = name
    op.inputs.append(qtensor)
    op.outputs.append(ftensor)
    op.setParsed()

    for c in qtensor.consumers:
        c.replaceInput(qtensor, ftensor)
        ftensor.addConsumer(c)
    ftensor.addProducer(op)
    qtensor.addConsumer(op)

    return (ftensor, op)
