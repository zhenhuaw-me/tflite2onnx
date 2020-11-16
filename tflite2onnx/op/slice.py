import copy
import logging
import tflite
import numpy as np

from tflite2onnx.op.common import Operator

logger = logging.getLogger('tflite2onnx')


class Slice(Operator):
    TypeMapping = {
        tflite.BuiltinOperator.STRIDED_SLICE: 'Slice',
    }

    def __init__(self, TFactory, index):
        super().__init__(TFactory, index)
        self.setInited()

    @property
    def type(self):
        return 'Slice'

    def parse(self):
        logger.debug("Parsing %s...", self.type)

        op = self.tflite
        opcode = self.model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
        assert(opcode in self.TypeMapping)

        assert(op.InputsLength() == 4)
        assert(op.OutputsLength() == 1)

        # input
        it = self.parseInput(0)
        rank = len(it.shape)

        # output
        ot = self.parseOutput(0)
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
        bt = self.parseInput(1)
        assert(bt.isInitializer)
        assert(rank == bt.shape[0])
        for i, (mask, begin) in enumerate(zip(m_begin, list(bt.data))):
            bt.data[i] = 0 if mask == 1 else begin

        # end
        et = self.parseInput(2)
        assert(et.isInitializer)
        assert(rank == et.shape[0])
        for i, (extent, mask, end) in enumerate(zip(it.shape, m_end, list(et.data))):
            et.data[i] = extent if mask == 1 else end

        # axis, we create from empty
        axis = np.arange(rank)
        at = self.TFactory.createVector(axis.astype('int32'))
        at.addConsumer(self)
        self.inputs.append(at)

        # strides
        st = self.parseInput(3)
        assert(st.isInitializer)
        assert(rank == st.shape[0])

        self.setParsed()

    def propagatableTensors(self):
        return [self.inputs[0], self.outputs[0]]

    def transform(self):
        logger.debug("Transforming %s...", self.shorty)
        layout = self.outputs[0].layout
        cl = copy.deepcopy(layout)
        if cl is None:
            logger.warning("layout of %s should not be None", self.shorty)
            return
        assert(len(self.inputs) == 5)
        tbegin = self.inputs[1]
        tbegin.data = cl.transform(tbegin.data)
        tend = self.inputs[2]
        tend.data = cl.transform(tend.data)
        tstrides = self.inputs[4]
        tstrides.data = cl.transform(tstrides.data)
