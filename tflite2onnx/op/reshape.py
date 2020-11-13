import copy
import logging
import tflite
import numpy as np

from tflite2onnx.op.transpose import Transpose
from tflite2onnx import mapping
from tflite2onnx.op.common import Operator

logger = logging.getLogger('tflite2onnx')


class Reshape(Operator):
    TypeMapping = {
        tflite.BuiltinOperator.RESHAPE: 'Reshape',
    }

    def __init__(self, TFactory, index):
        super().__init__(TFactory, index)

        self.forFakeBroadcasting = False

        self.setInited()

    @property
    def type(self):
        return 'Reshape'

    def parse(self):
        logger.debug("Parsing %s...", self.type)

        op = self.tflite
        opcode = self.model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
        assert(opcode is tflite.BuiltinOperator.RESHAPE)

        assert(op.InputsLength() >= 1)
        assert(op.OutputsLength() == 1)

        # input
        data = self.parseInput(0)

        if op.InputsLength() == 1:
            # options
            op_opt = op.BuiltinOptions()
            option = tflite.ReshapeOptions()
            option.Init(op_opt.Bytes, op_opt.Pos)
            sp = option.NewShapeAsNumpy()
            sp = self.TFactory.createVector(data, sp)
            sp.addConsumer(self)
            sp.dtype = mapping.DTYPE_NAME2ONNX['int64']
            self.inputs.append(sp)

        if op.InputsLength() == 2:
            # shape
            st = self.parseInput(1)

            # TFLite shape is int32 data type, ONNX is int64
            st.dtype = mapping.DTYPE_NAME2ONNX['int64']
            if st.isInitializer:
                st.data = st.data.astype('int64')
            if len(st.shape) > 1:
                logger.warning("ONNXRuntime doesn't support 2+rank shape, "
                               "flatten if the shape is initialzier, ignore otherwise."
                               "https://github.com/jackwish/tflite2onnx/issues/16")
                if st.isInitializer:
                    st.shape = (np.prod(np.array(st.shape)),)

        # output
        self.parseOutput(0)

        self.fakeTranspose()
        self.setParsed()

    def fakeTranspose(self):
        # Binary operators need to broadcast shape explicitly here since
        # they may not be broadcastable after layout propagration.
        # We don't really broadcast here, but extend shape with 1.
        assert(self.status.initialized)
        todo = self.inputs[0]

        new_t_name = 'TFLITE2ONNX_Transpose_%s' % todo.name
        new_t = self.TFactory.getWithRef(todo, new_t_name, True)
        new_t.shape = todo.shape
        new_t.asDtype('float32')
        print("todo shape", todo.shape)
        new_t.setParsed()

        trans = Transpose(self.TFactory, -1)
        shape_t_name = 'TFLITE2ONNX_Perm_%s' % todo.name
        self.attrs['perm'] = self.TFactory.getWithRef(todo, shape_t_name, True)
        self.attrs['perm'].asDtype('int64')

        trans.inputs.append(todo)
        todo.replaceConsumer(self, trans)
        self.replaceInput(todo, new_t)

        trans.outputs.append(new_t)
        new_t.addProducer(trans)
        new_t.addConsumer(self)
        trans.setParsed()

        self.pre.append(trans)

    def propagatableTensors(self):
        return list()

    def transform(self):
        if not self.forFakeBroadcasting:
            return
        i = self.inputs[0]
        o = self.outputs[0]
        assert(len(i.shape) != len(o.shape))
        shape_t = self.inputs[1]
        layout = copy.deepcopy(o.layout)
        if layout is None:
            raise ValueError("Requires layout description for <%s>" % i.name)
        shape_t.data = layout.transform(shape_t.data)
