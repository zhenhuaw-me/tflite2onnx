import copy
import logging
import tflite
import numpy as np

from tflite2onnx import mapping
from tflite2onnx.op.common import Operator
from tflite2onnx.op.transpose import Transpose

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
        self.parseInput(0)

        # This block has been commented
        # because the `Reshape` with only one input seems like a special case
        # haven't manage to reproduce currently
        # data = self.parseInput(0)
        if op.InputsLength() == 1:
            # options
            op_opt = op.BuiltinOptions()
            option = tflite.ReshapeOptions()
            option.Init(op_opt.Bytes, op_opt.Pos)
            sp = option.NewShapeAsNumpy()
            sp = self.TFactory.createVector(sp.astype('int64'))
            sp.addConsumer(self)
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

        # Insert a `Transpose`
        self.fakeTranspose()

        self.setParsed()

    def fakeTranspose(self):
        # If the input (or output) tensor of Reshape
        # holds a special data layout semantic,
        # we need to insert Transpose before (or after) the Reshape operator
        # https://github.com/jackwish/tflite2onnx/issues/28

        assert(self.status.initialized)
        to_transpose = self.inputs[0]

        transposed_name = 'TFLITE2ONNX_Transposed_%s' % to_transpose.name
        transposed = self.TFactory.getWithRef(to_transpose, transposed_name, True)
        transposed.shape = [to_transpose.shape[p] for p in [0, 2, 3, 1]]
        transposed.setParsed()

        # Construct the additional transpose
        trans = Transpose(self.TFactory, -1)
        trans.attrs['perm'] = np.array([0, 2, 3, 1])

        trans.inputs.append(to_transpose)
        to_transpose.replaceConsumer(self, trans)
        self.replaceInput(to_transpose, transposed)

        trans.outputs.append(transposed)
        transposed.addProducer(trans)
        transposed.addConsumer(self)
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
