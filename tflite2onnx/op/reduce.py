import logging
import tflite
from onnx import helper

from tflite2onnx import tensor
from tflite2onnx.op.operator import Operator

logger = logging.getLogger('tflite2onnx')


OpTypeMapping = {
    tflite.BuiltinOperator.MEAN: 'ReduceMean',
}


class Reduce(Operator):
    def __init__(self, model, graph, index):
        super().__init__(model, graph, index)

        self.axes = None
        self.keepdims = 0

        self.setInited()

    @property
    def type(self):
        if self.status.uninitialized:
            return 'Reduce'
        else:
            op = self.tflite
            opcode = self.model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
            assert(opcode in OpTypeMapping)
            return OpTypeMapping[opcode]

    @property
    def layoutPropagatable(self):
        return False

    def parse(self):
        logger.debug("Parsing %s...", self.type)

        op = self.tflite
        opcode = self.model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
        assert(opcode in OpTypeMapping)

        assert(op.InputsLength() == 2)
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

        # options
        ai = op.Inputs(1)
        self.axes = tensor.getData(self.model, self.graph, ai, 'int32')
        self.keepdims = 1 if (len(ot.shape) == len(it.shape)) else 0

        self.setParsed()

    def transform(self):
        layout = self.inputs[0].layout
        if layout is None:
            return
        else:
            axes = [axe if axe >= 0 else (self.axes + len(layout.perm)) for axe in self.axes]
            self.axes = [layout.perm.index(axe) for axe in axes]

    def convert(self):
        logger.debug("Converting %s...", self.type)
        self._convertTensors()

        inames = [t.name for t in self.inputs]
        onames = [t.name for t in self.outputs]
        self.onnx = helper.make_node(self.type, inames, onames,
                                     axes=self.axes, keepdims=self.keepdims)
        self.setConverted()
