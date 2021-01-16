import copy
import logging
import tflite
from onnx import helper

from tflite2onnx.tensor import TensorFactory
from tflite2onnx.common import T2OBase
from tflite2onnx.layout import Layout
from tflite2onnx.op import OpFactory
from tflite2onnx.quantize import handleQuantizationTensor
from tflite2onnx.quantize import foldFP16QuantPattern

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
        self.TFactory = TensorFactory(model, graph)
        self.OPCFactory = OpFactory(self.TFactory)

        self.setInited()

    def _collectOpAndTensor(self):
        self.op_all.clear()

        # collect operators
        def _recursive(op):
            for cur_op in op.pre:
                _recursive(cur_op)
            self.op_all.append(op)
            for cur_op in op.post:
                _recursive(cur_op)
        for op in self.ops:
            _recursive(op)

        # collect tensors
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
            logger.debug("Parsing operator: %d", i)
            op = self.OPCFactory.create(i)
            op.parse()
            self.ops.append(op)

        # inputs
        for i in range(self.graph.InputsLength()):
            # FIXME: assert they have been created.
            index = self.graph.Inputs(i)
            t = self.TFactory.get(index)
            self.inputs.append(t)

        # outputs
        for i in range(self.graph.OutputsLength()):
            index = self.graph.Outputs(i)
            t = self.TFactory.get(index)
            self.outputs.append(t)

        self._collectOpAndTensor()

        self.setParsed()

    def validate(self):
        self._collectOpAndTensor()
        for op in self.op_all:
            op.validate()
        for t in self.initializer | self.value_info:
            t.validate()

    def convert(self, explicit_layouts):
        logger.debug("Converting...")

        logger.debug("Handling data layout...")
        for op in self.ops:
            for t in op.inputs + op.outputs:
                if t.name in explicit_layouts:
                    assert(t.layout is None)
                    layouts = explicit_layouts[t.name]
                    assert(len(layouts) == 2)
                    t.layout = Layout(layouts[0], layouts[1])
        self._propagateLayout()
        self._collectOpAndTensor()

        foldFP16QuantPattern(self.ops)
        self._collectOpAndTensor()

        logger.debug("Translating quantization semantic...")
        for t in self.value_info | self.initializer:
            deqt = handleQuantizationTensor(self.TFactory, t)
            for i, o in enumerate(self.outputs):
                if o == t:
                    self.outputs[i] = deqt
        self._collectOpAndTensor()

        logger.debug("Graph:\n%s", str(self))

        self.validate()
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
        logger.debug("Propagation: %d tensors in total, %d to walk, %d at wild",
                     tensor_count, len(T_toWalk), len(T_wild))

        # propagrate layout across graph
        T_ignored = set()
        T_walked = set()
        while (len(T_toWalk) != 0):
            T = T_toWalk.pop()
            logger.debug("Propagation: walking %s", T.shorty)
            for n in T.producers + T.consumers:
                for t in n.propagatableTensors():
                    if t is T:
                        continue
                    if t in T_wild:
                        logger.debug("Propagation: propagated to %s", t.shorty)
                        assert(t.layout is None)
                        T_wild.remove(t)
                        if t.isScalar:
                            T_ignored.add(t)
                        else:
                            t.layout = copy.deepcopy(T.layout)
                            T_toWalk.add(t)
            T_walked.add(T)
        logger.debug("Propagation: wild tensors %d, ignored tensors %d",
                     len(T_wild), len(T_ignored))

        # update tensor and operator
        for t in T_walked:
            t.transform()
        self._collectOpAndTensor()
        for op in self.op_all:
            op.transform()

    def _dump(self, tag, container, useShorty):
        dump = str()
        for e in container:
            dump += '[%s] %s\n' % (tag, e.shorty if useShorty else e)
        return dump

    @property
    def shorty(self):
        string = str()
        string += self._dump('OP', self.op_all, True)
        string += self._dump('Input', self.inputs, True)
        string += self._dump('Output', self.outputs, True)
        string += self._dump('Initializer', self.initializer, True)
        string += self._dump('Value Info', self.value_info, True)
        return string

    def __str__(self):
        string = str()
        string += self._dump('OP', self.op_all, False)
        string += self._dump('Input', self.inputs, False)
        string += self._dump('Output', self.outputs, False)
        string += self._dump('Initializer', self.initializer, False)
        string += self._dump('Value Info', self.value_info, False)
        return string
