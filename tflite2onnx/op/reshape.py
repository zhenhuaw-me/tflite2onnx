import copy
import logging
import tflite
import numpy as np

from tflite2onnx import mapping
from tflite2onnx import tensor
from tflite2onnx.op.operator import Operator

logger = logging.getLogger('tflite2onnx')


class Reshape(Operator):
    def __init__(self, model, graph, index):
        super().__init__(model, graph, index)

        self.forFakeBroadcasting = False

        self.setInited()

    @property
    def type(self):
        return 'Reshape'

    @property
    def layoutPropagatable(self):
        return False

    def parse(self):
        logger.debug("Parsing %s...", self.type)

        op = self.tflite
        opcode = self.model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
        assert(opcode is tflite.BuiltinOperator.RESHAPE)

        assert(op.InputsLength() == 2)
        assert(op.OutputsLength() == 1)

        # input
        ii = op.Inputs(0)
        it = tensor.get(self.model, self.graph, ii)
        it.parse()
        it.addConsumer(self)
        self.inputs.append(it)

        # shape
        si = op.Inputs(1)
        st = tensor.get(self.model, self.graph, si)
        st.parse()
        # TFLite shape is int32 data type, ONNX is int64
        st.dtype = mapping.DTYPE_NAME2ONNX['int64']
        st.addConsumer(self)
        self.inputs.append(st)
        if st.isInitializer:
            st.data = st.data.astype('int64')
        if (len(st.shape) > 1):
            logger.warning("ONNXRuntime doesn't support 2+rank shape, "
                           "flatten if the shape is initialzier, ignore otherwise."
                           "https://github.com/jackwish/tflite2onnx/issues/16")
            if st.isInitializer:
                st.shape = (np.prod(np.array(st.shape)),)

        # output
        oi = op.Outputs(0)
        ot = tensor.get(self.model, self.graph, oi)
        ot.parse()
        ot.addProducer(self)
        self.outputs.append(ot)

        self.setParsed()

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
