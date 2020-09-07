import logging
import tflite

from tflite2onnx import tensor
from tflite2onnx.op.activation import handleFusedActivation
from tflite2onnx.op.operator import Operator

logger = logging.getLogger('tflite2onnx')


class FullyConnected(Operator):
    def __init__(self, model, graph, index):
        super().__init__(model, graph, index)

        # raw default values
        self.attrs['alpha'] = 1.0
        self.attrs['beta'] = 1.0
        # TFLite Fully Connected: A (M, K), B (N, K)
        self.attrs['transA'] = 0
        self.attrs['transB'] = 1

        self.setInited()

    @property
    def type(self):
        return 'Gemm'

    def parse(self):
        logger.debug("Parsing %s...", self.type)
        op = self.tflite
        opcode = self.model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
        assert(opcode is tflite.BuiltinOperator.FULLY_CONNECTED)

        self.has_bias = op.InputsLength() == 3
        assert(self.has_bias), "TFLite Fullly Connected always has bias"
        assert(op.OutputsLength() == 1)

        # input
        ii = op.Inputs(0)
        it = tensor.get(self.model, self.graph, ii)
        it.parse()
        it.addConsumer(self)
        self.inputs.append(it)

        # weight
        wi = op.Inputs(1)
        wt = tensor.get(self.model, self.graph, wi)
        wt.parse()
        wt.addConsumer(self)
        self.inputs.append(wt)

        # output
        oi = op.Outputs(0)
        ot = tensor.get(self.model, self.graph, oi)
        ot.parse()
        ot.addProducer(self)
        self.outputs.append(ot)

        # bias
        if self.has_bias:
            bi = op.Inputs(2)
            bt = tensor.get(self.model, self.graph, bi, is_bias=True)
            bt.parse()
            bt.addConsumer(self)
            self.inputs.append(bt)

        # options
        op_opt = op.BuiltinOptions()
        option = tflite.FullyConnectedOptions()
        option.Init(op_opt.Bytes, op_opt.Pos)

        assert(not option.KeepNumDims())
        assert(option.WeightsFormat() is tflite.FullyConnectedOptionsWeightsFormat.DEFAULT)

        handleFusedActivation(self, option, ot)

        self.setParsed()

    def propagatableTensors(self):
        return list()

    def transform(self):
        pass
