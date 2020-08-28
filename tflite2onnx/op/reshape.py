import copy
import logging
import tflite
from onnx import helper

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
        st = tensor.get(self.model, self.graph, si, None, True)
        st.parse()
        # TFLite shape is int32 data type, ONNX is int64
        st.dtype = mapping.DTYPE_NAME2ONNX['int64']
        st.data = st.data.astype('int64')
        st.addConsumer(self)
        self.inputs.append(st)
        logger.warning("In Reshape parsing, may need to handle new shape regarding layout.")

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

    def convert(self):
        logger.debug("Converting %s...", self.type)

        for t in self.inputs:
            t.convert()
        self.outputs[0].convert()

        inames = [t.name for t in self.inputs]
        onames = [t.name for t in self.outputs]
        self.onnx = helper.make_node(self.type, inames, onames)
        self.setConverted()
