import copy
import logging
import tflite
import numpy as np

from tflite2onnx import mapping
from tflite2onnx.op.common import Operator
from tflite2onnx.op.activation import handleFusedActivation
from tflite2onnx.op.reshape import Reshape

logger = logging.getLogger('tflite2onnx')


class Binary(Operator):
    TypeMapping = {
        tflite.BuiltinOperator.ADD: 'Add',
        tflite.BuiltinOperator.MUL: 'Mul',
        tflite.BuiltinOperator.SUB: 'Sub',
        tflite.BuiltinOperator.POW: 'Pow',
    }

    OptionMapping = {
        tflite.BuiltinOperator.ADD: tflite.AddOptions,
        tflite.BuiltinOperator.MUL: tflite.MulOptions,
        tflite.BuiltinOperator.SUB: tflite.SubOptions,
    }

    def __init__(self, TFactory, index):
        super().__init__(TFactory, index)
        self.setInited()

    @property
    def type(self):
        if self.status.uninitialized:
            return 'Binary'
        else:
            op = self.tflite
            opcode = self.model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
            assert(opcode in self.TypeMapping)
            return self.TypeMapping[opcode]

    def fakeBroadcast(self):
        # Binary operators need to broadcast shape explicitly here since
        # they may not be broadcastable after layout propagration.
        # We don't really broadcast here, but extend shape with 1.
        assert(self.status.initialized)
        a = self.inputs[0]
        b = self.inputs[1]
        output = self.outputs[0]
        if (len(a.shape) == len(b.shape)):
            return
        logger.info("Inserting `Reshape` for fake broadcasting, be carefull for the layout")

        align_a, new_shape = alignDimension(a.shape, b.shape)
        todo = a if align_a else b
        assert(len(new_shape) == len(output.shape))

        new_t_name = 'TFLITE2ONNX_Reshape_%s' % todo.name
        new_t = self.TFactory.getWithRef(todo, new_t_name, True)
        new_t.shape = new_shape
        new_t.setParsed()

        shape_t_name = 'TFLITE2ONNX_NewShape_%s' % todo.name
        shape_t = self.TFactory.getWithRef(todo, shape_t_name, True)
        shape_t.dtype = mapping.DTYPE_NAME2ONNX['int64']
        shape_t.shape = (len(new_shape),)
        shape_t.data = np.array(new_shape)
        shape_t.setParsed()

        reshape = Reshape(self.TFactory, -1)
        reshape.forFakeBroadcasting = True

        reshape.inputs.append(todo)
        todo.replaceConsumer(self, reshape)
        self.replaceInput(todo, new_t)

        reshape.inputs.append(shape_t)
        shape_t.addConsumer(reshape)

        reshape.outputs.append(new_t)
        new_t.addProducer(reshape)
        new_t.addConsumer(self)
        reshape.setParsed()

        self.pre.append(reshape)

    def parse(self):
        logger.debug("Parsing %s...", self.type)

        op = self.tflite
        opcode = self.model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
        assert(opcode in self.TypeMapping)

        assert(op.InputsLength() == 2)
        assert(op.OutputsLength() == 1)

        self.parseInput(0)
        self.parseInput(1)
        ot = self.parseOutput(0)

        self.fakeBroadcast()

        # options
        op_opt = op.BuiltinOptions()
        if opcode in self.OptionMapping:
            option = self.OptionMapping[opcode]()
            option.Init(op_opt.Bytes, op_opt.Pos)

            handleFusedActivation(self, option, ot)

        self.setParsed()

    def propagatableTensors(self):
        return self.inputs + self.outputs

    def transform(self):
        pass


def alignDimension(a, b):
    """Align the dimension of the shorter one to the longer one.

    We don't really broadcast tensors during converting, instead, align
    dimensions of the two inputs such that the tensors have same dimensions
    which is _layout handling compatible_.
    """
    align_a = len(a) < len(b)
    to_align = a if align_a else b
    ref = b if align_a else a

    size = len(ref) - len(to_align)
    aligned = copy.deepcopy(to_align)
    for i in range(size):
        aligned.insert(0, 1)

    return (align_a, aligned)


# wrapper is used here to override Binary.type property
class PowerWrapper(Binary):
    @property
    def type(self):
        return 'Pow'
