import copy
import logging
import tflite
import numpy as np

from tflite2onnx import mapping
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

    def parse(self):
        logger.debug("Parsing %s...", self.type)

        op = self.tflite
        opcode = self.model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
        assert(opcode is tflite.BuiltinOperator.RESHAPE)

        assert(op.InputsLength() == 2)
        assert(op.OutputsLength() == 1)

        # input
        self.parseInput(0)

        # shape
        st = self.parseInput(1)
        # TFLite shape is int32 data type, ONNX is int64
        st.dtype = mapping.DTYPE_NAME2ONNX['int64']
        if st.isInitializer:
            st.data = st.data.astype('int64')
        if (len(st.shape) > 1):
            logger.warning("ONNXRuntime doesn't support 2+rank shape, "
                           "flatten if the shape is initialzier, ignore otherwise."
                           "https://github.com/jackwish/tflite2onnx/issues/16")
            if st.isInitializer:
                st.shape = (np.prod(np.array(st.shape)),)

        # output
        self.parseOutput(0)

        self.setParsed()

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
