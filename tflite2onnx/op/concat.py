import logging
import tflite
from onnx import helper

from tflite2onnx import tensor
from tflite2onnx.op.operator import Operator
from tflite2onnx.op.activation import handleFusedActivation
from tflite2onnx.layout import Layout

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
    def sensitive(self):
        return True

    def parse(self):
        logger.debug("Parsing %s...", self.type)
        op = self.tflite
        opcode = self.model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
        assert(opcode is tflite.BuiltinOperator.CONCATENATION)

        assert(op.InputsLength() >= 1)
        assert(op.OutputsLength() == 1)

        # ii = op.Inputs(0)
        # ilayout = Layout('NHWC', 'NCHW')
        # it = tensor.get(self.model, self.graph, ii, ilayout)
        # it.parse()
        # it.addConsumer(self)
        # self.inputs.append(it)

        # op_opt = op.BuiltinOptions()
        # option = tflite.Pool2DOptions()
        # option.Init(op_opt.Bytes, op_opt.Pos)
        # self.axis = option.Padding()]

        # oi = op.Outputs(0)
        # olayout = Layout('NHWC', 'NCHW')
        # ot = tensor.get(self.model, self.graph, oi, olayout)
        # ot.parse()

        # handleFusedActivation(self, option, ot)

        self.setParsed()

    def convert(self):
        logger.debug("Converting %s...", self.type)

        for t in self.inputs:
            t.convert()
        self.outputs[0].convert()

        inames = [t.name for t in self.inputs]
        onames = [t.name for t in self.outputs]
        logger.debug("Making ONNX...")
        self.onnx = helper.make_node(self.type, inames, onames, axis=self.axis)
