import logging
import tflite

from tflite2onnx import tensor
from tflite2onnx.op.operator import Operator

logger = logging.getLogger('tflite2onnx')


class Softmax(Operator):
    def __init__(self, model, graph, index):
        super().__init__(model, graph, index)

        self.attrs['axis'] = -1

        self.setInited()

    @property
    def type(self):
        return 'Softmax'

    def parse(self):
        logger.debug("Parsing %s...", self.type)
        op = self.tflite
        opcode = self.model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
        assert(opcode is tflite.BuiltinOperator.SOFTMAX)

        assert(op.InputsLength() == 1)
        assert(op.OutputsLength() == 1)

        ii = op.Inputs(0)
        it = tensor.get(self.model, self.graph, ii)
        it.parse()
        it.addConsumer(self)
        self.inputs.append(it)

        # TFLite Softmax ALWAYS softmax on `-1` axis, while ONNX on `1` by default.
        # And, we transform NHWC to NCHW for 4D tensor.
        # axis = 1 if len(to.shape) == 4 else -1
        # if len(to.shape) == 4:
        #     axis = 1
        # elif len(to.shape) == 2:
        #     axis = -1
        # else:
        #     axis = -1
        #     logger.warning("Softmax has input shape %s.", str(to.shape))
        # FIXME: axis is the dim index of 'C'.
        self.attrs['axis'] = -1

        oi = op.Outputs(0)
        ot = tensor.get(self.model, self.graph, oi)
        ot.parse()
        ot.addProducer(self)
        self.outputs.append(ot)

        self.setParsed()

    def propagatableTensors(self):
        return self.inputs + self.outputs

    def transform(self):
        pass
