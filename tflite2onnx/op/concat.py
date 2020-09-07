import logging
import tflite

from tflite2onnx import tensor
from tflite2onnx.op.activation import handleFusedActivation
from tflite2onnx.op.operator import Operator

logger = logging.getLogger('tflite2onnx')


class Concat(Operator):
    def __init__(self, model, graph, index):
        super().__init__(model, graph, index)

        self.attrs['axis'] = -1

        self.setInited()

    @property
    def type(self):
        return 'Concat'

    def parse(self):
        logger.debug("Parsing %s...", self.shorty)
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
        self.attrs['axis'] = option.Axis()

        oi = op.Outputs(0)
        ot = tensor.get(self.model, self.graph, oi)
        ot.parse()
        ot.addProducer(self)
        self.outputs.append(ot)

        handleFusedActivation(self, option, ot)

        self.setParsed()

    def propagatableTensors(self):
        return self.inputs + self.outputs

    def transform(self):
        logger.debug("Transforming %s...", self.shorty)
        layout = self.outputs[0].layout
        if layout is not None:
            axis = self.attrs['axis']
            axis = axis if axis >= 0 else (axis + len(layout.perm))
            self.attrs['axis'] = layout.perm.index(axis)
