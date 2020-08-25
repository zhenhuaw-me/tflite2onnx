import logging
import tflite
from onnx import helper

from tflite2onnx import tensor
from tflite2onnx.op.operator import Operator
from tflite2onnx.op.activation import handleFusedActivation

logger = logging.getLogger('tflite2onnx')


OpTypeMapping = {
    tflite.BuiltinOperator.ADD: 'Add',
    tflite.BuiltinOperator.MUL: 'Mul',
}

OpOptionFuncMapping = {
    tflite.BuiltinOperator.ADD: tflite.AddOptions,
    tflite.BuiltinOperator.MUL: tflite.MulOptions,
}


class Binary(Operator):
    def __init__(self, model, graph, index):
        super().__init__(model, graph, index)
        logger.debug("Converting...")
        self.setInited()

    @property
    def type(self):
        if self.status.uninitialized:
            return 'Binary'
        else:
            op = self.tflite
            opcode = self.model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
            assert(opcode in OpTypeMapping)
            return OpTypeMapping[opcode]

    @property
    def implicitLayout(self):
        return False

    def parse(self):
        logger.debug("Parsing %s...", self.type)

        op = self.tflite
        opcode = self.model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
        assert(opcode in OpTypeMapping)

        assert(op.InputsLength() == 2)
        assert(op.OutputsLength() == 1)

        for i in range(op.InputsLength()):
            ii = op.Inputs(i)
            it = tensor.get(self.model, self.graph, ii)
            it.parse()
            it.addConsumer(self)
            self.inputs.append(it)

        oi = op.Outputs(0)
        ot = tensor.get(self.model, self.graph, oi)
        ot.parse()
        ot.addProducer(self)
        self.outputs.append(ot)

        # options
        op_opt = op.BuiltinOptions()
        option = OpOptionFuncMapping[opcode]()
        option.Init(op_opt.Bytes, op_opt.Pos)

        handleFusedActivation(self, option, ot)

        self.setParsed()

    def transform(self):
        pass

    def convert(self):
        logger.debug("Converting %s...", self.type)

        for t in self.inputs:
            t.convert()
        self.outputs[0].convert()

        inames = [t.name for t in self.inputs]
        onames = [t.name for t in self.outputs]
        self.onnx = helper.make_node(self.type, inames, onames)
        self.setConverted()


def fakeBroadcast(a, b):
    extend_a = len(a) < len(b)
    to_extend = a if extend_a else b
    ref = b if extend_a else a

    size = len(ref) - len(to_extend)
    for i in range(size):
        to_extend.insert(0, 1)

    return (a, b)
