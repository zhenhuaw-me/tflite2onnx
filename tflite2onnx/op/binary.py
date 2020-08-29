import copy
import logging
import tflite
import numpy as np

from tflite2onnx import mapping
from tflite2onnx import tensor
from tflite2onnx.op.operator import Operator
from tflite2onnx.op.activation import handleFusedActivation
from tflite2onnx.op.reshape import Reshape

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
    def layoutPropagatable(self):
        return True

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

        new_t = tensor.Tensor(todo.model, todo.graph, -1)
        new_t.name = 'TFLITE2ONNX_Reshape_%s' % todo.name
        new_t.dtype = todo.dtype
        new_t.scale = copy.deepcopy(todo.scale)
        new_t.zero_point = copy.deepcopy(todo.zero_point)
        new_t.layout = copy.deepcopy(todo.layout)
        new_t.shape = new_shape
        new_t.setParsed()
        assert(new_t.name not in tensor.registery)
        tensor.registery[new_t.name] = new_t

        shape_t = tensor.Tensor(todo.model, todo.graph, -1)
        shape_t.name = 'TFLITE2ONNX_NewShape_%s' % todo.name
        shape_t.dtype = mapping.DTYPE_NAME2ONNX['int64']
        shape_t.shape = (len(new_shape),)
        shape_t.data = np.array(new_shape)
        shape_t.setParsed()

        reshape = Reshape(todo.model, todo.graph, -1)
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

        self.fakeBroadcast()

        # options
        op_opt = op.BuiltinOptions()
        option = OpOptionFuncMapping[opcode]()
        option.Init(op_opt.Bytes, op_opt.Pos)

        handleFusedActivation(self, option, ot)

        self.setParsed()

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
