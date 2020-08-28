import copy
import logging
import tflite
from onnx import helper

from tflite2onnx import tensor
from tflite2onnx.common import T2OBase
from tflite2onnx.layout import Layout
from tflite2onnx.op import getOp
from tflite2onnx.quantize import handleQuantizationTensor

logger = logging.getLogger('tflite2onnx')


class Graph(T2OBase):
    def __init__(self, model: tflite.Model, graph: tflite.SubGraph):
        super().__init__(model, graph)
        self.ops = []   # the OP that has TFLite peer
        self.op_all = []  # includes helper OP
        self.inputs = []
        self.outputs = []
        self.initializer = set()
        self.value_info = set()
        self.tflite = graph
        tensor.registery.clear()
        self.setInited()

    def _collectOpAndTensor(self):
        self.op_all.clear()

        def _recursive(op):
            for cur_op in op.pre:
                _recursive(cur_op)
            self.op_all.append(op)
            for cur_op in op.post:
                _recursive(cur_op)
        for op in self.ops:
            _recursive(op)

        assert(len(self.op_all) > 0)
        self.initializer.clear()
        self.value_info.clear()
        for op in self.op_all:
            for t in op.inputs + op.outputs:
                if t.isInitializer:
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
            self.ops.append(op)

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

        self._collectOpAndTensor()

        self.setParsed()

    def convert(self, explicit_layouts):
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
        self._propagateLayout()
        self._collectOpAndTensor()

        logger.debug("Translating quantization pattern...")
        for t in self.value_info | self.initializer:
            deqt = handleQuantizationTensor(t)
            for i, o in enumerate(self.outputs):
                if o == t:
                    self.outputs[i] = deqt
        self._collectOpAndTensor()

        logger.debug("Graph:\n%s", str(self))

        for op in self.op_all:
            op.convert()

        logger.debug("Making ONNX...")
        onodes = [n.onnx for n in self.op_all]
        oinputs = [t.onnx for t in self.inputs]
        ooutputs = [t.onnx for t in self.outputs]
        initializer = [t.onnx for t in self.initializer]
        value_info = [t.onnx for t in self.value_info]

        self.onnx = helper.make_graph(onodes, 'pre-alpha', oinputs, ooutputs,
                                      initializer=initializer, value_info=value_info)
        self.setConverted()

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
            logger.debug("Propagation: walking %s" % str(T))
            for n in T.producers + T.consumers:
                if n.layoutPropagatable:
                    for t in n.inputs + n.outputs:
                        if t is T:
                            continue
                        if t in T_wild:
                            logger.debug("Propagation: propagated to %s" % str(t))
                            assert(t.layout is None)
                            T_wild.remove(t)
                            if t.isScalar:
                                T_ignored.add(t)
                            else:
                                t.layout = copy.deepcopy(T.layout)
                                T_toWalk.add(t)
                else:
                    for t in n.inputs + n.outputs:
                        # Bias has no layout information, and unneed to handle
                        if t in T_wild:
                            T_wild.remove(t)
                            T_ignored.add(t)
            T_walked.add(T)
        logger.debug("Propagation: wild tensors %d, ignored tensors %d" %
                     (len(T_wild), len(T_ignored)))

        # update tensor and operator
        for t in T_walked:
            t.transform()
        self._collectOpAndTensor()
        for op in self.op_all:
            op.transform()

    def __str__(self):
        string = str()
        for op in self.op_all:
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
