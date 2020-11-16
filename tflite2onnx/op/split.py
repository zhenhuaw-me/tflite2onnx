import logging
import tflite
import numpy as np

from tflite2onnx.tensor import TensorFactory
from tflite2onnx.op.common import Operator

logger = logging.getLogger('tflite2onnx')


class Split(Operator):
    TypeMapping = {
        tflite.BuiltinOperator.SPLIT: 'Split',
    }

    def __init__(self, TFactory, index):
        super().__init__(TFactory, index)

        self.attrs['axis'] = -1
        self.attrs['split'] = None

        self.setInited()

    @property
    def type(self):
        return 'Split'

    def parse(self):
        logger.debug("Parsing %s...", self.type)

        op = self.tflite
        opcode = self.model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
        assert(opcode in self.TypeMapping)

        assert(op.InputsLength() == 2)

        # input
        it = self.parseInput(1)

        # options
        ai = op.Inputs(0)
        axis = TensorFactory.getData(self.model, self.graph, ai, 'int32')
        assert(axis.size == 1)
        self.attrs['axis'] = int(axis[0])

        op_opt = op.BuiltinOptions()
        option = tflite.SplitOptions()
        option.Init(op_opt.Bytes, op_opt.Pos)
        # TFLite outputs have same shape
        split_size = option.NumSplits()
        assert(isinstance(split_size, int))
        assert(op.OutputsLength() == split_size)
        self.attrs['split'] = np.zeros(split_size).astype('int')

        # XXX workaround for ONNXRuntime: doesn't support all-zero `split`.
        split = it.shape[self.attrs['axis']] / split_size
        self.attrs['split'] = np.full((split_size,), split).astype('int')

        # output
        for i in range(split_size):
            self.parseOutput(i)

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
