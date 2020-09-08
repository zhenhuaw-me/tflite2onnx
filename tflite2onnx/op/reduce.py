import logging
import tflite

from tflite2onnx import tensor
from tflite2onnx.op.operator import Operator

logger = logging.getLogger('tflite2onnx')


OpTypeMapping = {
    tflite.BuiltinOperator.MEAN: 'ReduceMean',
}


class Reduce(Operator):
    def __init__(self, model, graph, tregistry, index):
        super().__init__(model, graph, tregistry, index)

        self.attrs['axes'] = None
        self.attrs['keepdims'] = 0

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

    def parse(self):
        logger.debug("Parsing %s...", self.type)

        op = self.tflite
        opcode = self.model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
        assert(opcode in OpTypeMapping)

        assert(op.InputsLength() == 2)
        assert(op.OutputsLength() == 1)
        it = self.parseInput(0)
        ot = self.parseOutput(0)

        # options
        ai = op.Inputs(1)
        self.attrs['axes'] = tensor.getData(self.model, self.graph, ai, 'int32')
        self.attrs['keepdims'] = 1 if (len(ot.shape) == len(it.shape)) else 0

        self.setParsed()

    def propagatableTensors(self):
        return list()

    def transform(self):
        layout = self.inputs[0].layout
        if layout is None:
            return
        else:
            axes = self.attrs['axes']
            axes = [axe if axe >= 0 else (axes + len(layout.perm)) for axe in axes]
            self.attrs['axes'] = [layout.perm.index(axe) for axe in axes]
