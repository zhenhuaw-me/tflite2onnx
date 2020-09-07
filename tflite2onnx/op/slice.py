import logging
import tflite
import numpy as np

from tflite2onnx import tensor
from tflite2onnx.op.operator import Operator

logger = logging.getLogger('tflite2onnx')


class Slice(Operator):
    def __init__(self, model, graph, index):
        super().__init__(model, graph, index)
        self.setInited()

    @property
    def type(self):
        return 'Slice'

    def parse(self):
        logger.debug("Parsing %s...", self.type)

        op = self.tflite
        opcode = self.model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
        assert(opcode is tflite.BuiltinOperator.STRIDED_SLICE)

        assert(op.InputsLength() == 4)
        assert(op.OutputsLength() == 1)

        # input
        ii = op.Inputs(0)
        it = tensor.get(self.model, self.graph, ii)
        it.parse()
        it.addConsumer(self)
        self.inputs.append(it)
        rank = len(it.shape)

        # output
        oi = op.Outputs(0)
        ot = tensor.get(self.model, self.graph, oi)
        ot.parse()
        ot.addProducer(self)
        self.outputs.append(ot)
        assert(rank == len(ot.shape))

        # options
        op_opt = op.BuiltinOptions()
        option = tflite.StridedSliceOptions()
        option.Init(op_opt.Bytes, op_opt.Pos)
        m_begin = option.BeginMask()
        m_end = option.EndMask()
        m_ellipsis = option.EllipsisMask()
        m_new_axis = option.NewAxisMask()
        m_shrink_axis = option.ShrinkAxisMask()
        assert(m_ellipsis == 0), "EllipsisMask not supported!"
        assert(m_new_axis == 0), "NewAxisMask not supported!"
        assert(m_shrink_axis == 0), "ShrinkAxisMask not supported!"

        def _intToBitsList(data, size):
            return [int(x) for x in '{:0{size}b}'.format(data, size=size)]

        m_begin = _intToBitsList(m_begin, rank)
        m_end = _intToBitsList(m_end, rank)

        # begin
        bi = op.Inputs(1)
        bt = tensor.get(self.model, self.graph, bi)
        bt.parse()
        assert(bt.isInitializer)
        assert(rank == bt.shape[0])
        for i, (mask, begin) in enumerate(zip(m_begin, list(bt.data))):
            bt.data[i] = 0 if mask == 1 else begin
        bt.addConsumer(self)
        self.inputs.append(bt)

        # end
        ei = op.Inputs(2)
        et = tensor.get(self.model, self.graph, ei)
        et.parse()
        assert(et.isInitializer)
        assert(rank == et.shape[0])
        for i, (extent, mask, end) in enumerate(zip(it.shape, m_end, list(et.data))):
            et.data[i] = extent if mask == 1 else end
        et.addConsumer(self)
        self.inputs.append(et)

        # axis, we create from empty
        axis = np.arange(rank)
        at = tensor.createVector(bt, axis)
        at.addConsumer(self)
        self.inputs.append(at)

        # strides
        si = op.Inputs(3)
        st = tensor.get(self.model, self.graph, si)
        st.parse()
        assert(st.isInitializer)
        assert(rank == st.shape[0])
        st.addConsumer(self)
        self.inputs.append(st)

        self.setParsed()

    def propagatableTensors(self):
        return [self.inputs[0], self.outputs[0]]

    def transform(self):
        logger.debug("Transforming %s...", self.shorty)
        layout = self.outputs[0].layout
        assert(layout is None)
        # if layout is not None:
