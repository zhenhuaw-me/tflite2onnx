import copy
import sys
import logging
import tflite
from onnx import helper

from tflite2onnx.common import T2OBase, LayoutApproach
from tflite2onnx import tensor
from tflite2onnx.op import getOp
from tflite2onnx.transpose import createTransposeHelper
from tflite2onnx.layout import Layout

logger = logging.getLogger('tflite2onnx')


class Graph(T2OBase):
    def __init__(self, model: tflite.Model, graph: tflite.SubGraph):
        super().__init__(model, graph)
        self.ops = []
        self.inputs = []
        self.outputs = []
        self.initializer = set()
        self.value_info = set()
        self.tflite = graph
        tensor.registery.clear()
        self.setInited()

    def _collectTensors(self):
        self.initializer.clear()
        self.value_info.clear()
        for op in self.ops:
            for t in op.inputs + op.outputs:
                if t.is_initializer:
                    self.initializer.add(t)
                else:
                    self.value_info.add(t)

    def parse(self):
        logger.debug("Parsing the Graph...")
        # operators
        for i in range(self.graph.OperatorsLength()):
            logger.debug("Parsing operator: {}".format(i))
            op = getOp(self.model, self.graph, i)
            op.parse()
            self.ops.extend(op.pre)
            self.ops.append(op)
            self.ops.extend(op.post)

        # inputs
        logger.debug("Parsing inputs...")
        for i in range(self.graph.InputsLength()):
            # FIXME: assert they have been created.
            index = self.graph.Inputs(i)
            t = tensor.get(self.model, self.graph, index)
            self.inputs.append(t)

        # outputs
        for i in range(self.graph.OutputsLength()):
            index = self.graph.Outputs(i)
            t = tensor.get(self.model, self.graph, index)
            self.outputs.append(t)

        self._collectTensors()

        self.setParsed()

    def convert(self, layout_approach, explicit_layouts):
        logger.debug("Converting...")

        logger.debug("Handling data layout...")

        for op in self.ops:
            tensors = op.inputs + op.outputs
            for t in tensors:
                if t.name in explicit_layouts:
                    assert(t.layout is None)
                    layouts = explicit_layouts[t.name]
                    assert(len(layouts) == 2)
                    t.layout = Layout(layouts[0], layouts[1])

        if layout_approach is LayoutApproach.TRANSPOSE:
            self._insertLayoutTranpose(explicit_layouts)
        elif layout_approach is LayoutApproach.PROPAGATION:
            # FIXME: explicit_layouts here?
            self._propagateLayout()
        else:
            assert(False), "Unkown LayoutApproach!"

        logger.debug("Dequantizing operators...")
        ops = [op for op in self.ops]
        for op in ops:
            logger.debug("Dequantizing %s...", str(op))
            op.dequantize()

        self._collectTensors()
        logger.debug("Graph:\n%s", str(self))

        for op in self.ops:
            op.convert()

        logger.debug("Making ONNX...")
        onodes = [n.onnx for n in self.ops]
        oinputs = [t.onnx for t in self.inputs]
        ooutputs = [t.onnx for t in self.outputs]
        initializer = [t.onnx for t in self.initializer]
        value_info = [t.onnx for t in self.value_info]

        self.onnx = helper.make_graph(onodes, 'pre-alpha', oinputs, ooutputs,
                                      initializer=initializer, value_info=value_info)
        self.setConverted()

    def _insertLayoutTranpose(self, explicit_layouts):        # noqa: C901
        logger.debug("Inserting Transpose to layout divergence node...")
        assert(self.status.parsed)
        # prepare
        op2index = dict()
        for index, op in enumerate(self.ops):
            op2index[op] = index

        def getMinIndex(ops, skip):
            ret = sys.maxsize
            for op in ops:
                if op is not skip:
                    ret = min(ret, op2index[op])
            return ret

        def getMaxIndex(ops, skip):
            ret = -1
            for op in ops:
                if op is not skip:
                    ret = max(ret, op2index[op])
            return ret
        op2insertIndex = list()  # collect where to insert Transpose op

        T_toWalk = list()
        for t in self.initializer | self.value_info:
            if not t.layoutMatch:
                T_toWalk.append(t)

        # walk and transpose all tensors
        for t in T_toWalk:
            isExplicitLayout = t.name in explicit_layouts

            def hasImplicitLayoutNode(ln):
                return any(n.implicitLayout for n in ln)

            if ((isExplicitLayout and (len(t.consumers) != 0)) or
                    hasImplicitLayoutNode(t.consumers)):
                logger.debug("transposing consumers for <%s>...", t.name)
                t2, transOp = createTransposeHelper(t, False)
                self.value_info.add(t2)
                ii = getMinIndex(t.consumers, transOp)
                op2insertIndex.append((transOp, ii))
                for op in t.consumers:
                    if op is not transOp:
                        op.replaceInput(t, t2)

            if ((isExplicitLayout and (len(t.producers) != 0)) or
                    hasImplicitLayoutNode(t.producers)):
                logger.debug("transposing producers for <%s>...", t.name)
                t2, transOp = createTransposeHelper(t, True)
                self.value_info.add(t2)
                ii = getMaxIndex(t.producers, transOp) + 1
                op2insertIndex.append((transOp, ii))
                for op in t.producers:
                    if op is not transOp:
                        op.replaceOutput(t, t2)

        # insert transpose op to graph
        op2insertIndex = sorted(op2insertIndex, key=lambda k: k[1], reverse=True)
        for op, index in op2insertIndex:
            self.ops.insert(index, op)

    def _propagateLayout(self):        # noqa: C901
        logger.debug("Propragating layout across graph...")

        # collect tensors
        T_toWalk = set()
        T_wild = set()
        tensor_count = len(self.value_info) + len(self.initializer)
        for t in self.value_info | self.initializer:
            if t.layout is None:
                T_wild.add(t)
            else:
                T_toWalk.add(t)
        logger.debug("Propagation: %d tensors in total, %d to walk, %d at wild" %
                     (tensor_count, len(T_toWalk), len(T_wild)))

        # propagrate layout across graph
        T_ignored = set()
        T_walked = set()
        while (len(T_toWalk) != 0):
            T = T_toWalk.pop()
            for n in T.producers + T.consumers:
                if n.implicitLayout:
                    for t in n.inputs + n.outputs:
                        # Bias has no layout information, and unneed to handle
                        if t in T_wild:
                            T_wild.remove(t)
                            T_ignored.add(t)
                else:
                    for t in n.inputs + n.outputs:
                        logger.debug("Propagation: processing %s" % str(t))
                        if t in T_wild:
                            assert(t.layout is None)
                            T_wild.remove(t)
                            if t.isScalar:
                                T_ignored.add(t)
                            else:
                                t.layout = copy.deepcopy(T.layout)
                                T_toWalk.add(t)
            T_walked.add(T)
        logger.debug("Propagation: wild tensors %d, ignored tensors %d" %
                     (len(T_wild), len(T_ignored)))

        # update tensor and operator
        for t in T_walked:
            t.transform()
        for op in self.ops:
            op.transform()

    def __str__(self):
        string = str()
        for op in self.ops:
            string += '[OP] ' + str(op) + '\n'
        for t in self.inputs:
            string += '[Inputs] ' + str(t) + '\n'
        for t in self.initializer:
            string += '[Initializer] ' + str(t) + '\n'
        for t in self.value_info:
            string += '[Value Info] ' + str(t) + '\n'
        for t in self.outputs:
            string += '[Outputs] ' + str(t) + '\n'
        return string
