import logging
import tflite
from onnx import helper

from tflite2onnx import tensor
from tflite2onnx.op.activation import handleFusedActivation
from tflite2onnx.op.operator import Operator

logger = logging.getLogger('tflite2onnx')


class Concat(Operator):
    def __init__(self, model, graph, index):
        super().__init__(model, graph, index)

        self.axis = -1

        self.setInited()

    @property
    def type(self):
        return 'Concat'

    @property
    def implicitLayout(self):
        return False

    @property
    def layoutPropagatable(self):
        return True

    def parse(self):
        logger.debug("Parsing %s...", self.type)
        op = self.tflite
        opcode = self.model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
        assert(opcode is tflite.BuiltinOperator.CONCATENATION)

        assert(op.InputsLength() >= 1)
        assert(op.OutputsLength() == 1)

        for i in range(op.InputsLength()):
            ii = op.Inputs(i)
            it = tensor.get(self.model, self.graph, ii)
            it.parse()
            it.addConsumer(self)
            self.inputs.append(it)

        op_opt = op.BuiltinOptions()
        option = tflite.ConcatenationOptions()
        option.Init(op_opt.Bytes, op_opt.Pos)
        self.axis = option.Axis()

        oi = op.Outputs(0)
        ot = tensor.get(self.model, self.graph, oi)
        ot.parse()
        ot.addProducer(self)
        self.outputs.append(ot)

        handleFusedActivation(self, option, ot)

        self.setParsed()

    def transform(self):
        layout = self.outputs[0].layout
        if layout is None:
            return
        else:
            axis = self.axis if self.axis >= 0 else (self.axis + len(layout.perm))
            self.axis = layout.perm.index(axis)

    def convert(self):
        logger.debug("Converting %s...", self.type)

        for t in self.inputs:
            t.convert()
        self.outputs[0].convert()

        inames = [t.name for t in self.inputs]
        onames = [t.name for t in self.outputs]
        logger.debug("Making ONNX...")
        self.onnx = helper.make_node(self.type, inames, onames, axis=self.axis)
