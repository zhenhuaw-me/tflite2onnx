import sys
import logging
import tflite
from onnx import helper

from tflite2onnx.common import T2OBase
from tflite2onnx import tensor
from tflite2onnx.op import getOp
from tflite2onnx.transpose import createTransposeHelper

logger = logging.getLogger('tflite2onnx')


class Graph(T2OBase):
    def __init__(self, model: tflite.Model, graph: tflite.SubGraph):
        super().__init__(model, graph)
        self.ops = []
        self.inputs = []
        self.outputs = []
        self.initializer = dict()
        self.value_info = dict()
        self.tflite = graph
        tensor.registery.clear()
        self.setInited()

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

        # collect tensors
        for op in self.ops:

            def uniqueInDict(t, d):
                return (t.name not in d) or (d[t.name] == t)

            tensors = op.inputs + op.outputs
            for t in tensors:
                if t.is_initializer:
                    assert(uniqueInDict(t, self.initializer))
                    self.initializer[t.name] = t
                else:
                    assert(uniqueInDict(t, self.value_info))
                    self.value_info[t.name] = t

        self.setParsed()

    def convert(self):
        logger.debug("Converting...")

        self._insertLayoutTranpose()

        logger.debug("Graph:\n%s", str(self))

        for op in self.ops:
            op.convert()

        logger.debug("Making ONNX...")
        onodes = [n.onnx for n in self.ops]
        oinputs = [t.onnx for t in self.inputs]
        ooutputs = [t.onnx for t in self.outputs]
        initializer = [t.onnx for n, t in self.initializer.items()]
        value_info = [t.onnx for n, t in self.value_info.items()]

        self.onnx = helper.make_graph(onodes, 'pre-alpha', oinputs, ooutputs,
                                      initializer=initializer, value_info=value_info)
        self.setConverted()

    def _insertLayoutTranpose(self):        # noqa: C901
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

        # walk and transpose all tensors
        all_tensors = {**self.initializer, **self.value_info}
        for _, t in all_tensors.items():
            if t.layoutMatch:
                continue
            logger.debug("<%s> layout not match", t.name)

            def hasSensitiveNode(ln):
                return any(n.sensitive for n in ln)

            if hasSensitiveNode(t.consumers):
                logger.debug("<%s> transposing consumers...", t.name)
                t2, transOp = createTransposeHelper(t, False)
                self.value_info[t2.name] = t2
                ii = getMinIndex(t.consumers, transOp)
                op2insertIndex.append((transOp, ii))
                for op in t.consumers:
                    if op is not transOp:
                        op.replaceInput(t, t2)

            if hasSensitiveNode(t.producers):
                logger.debug("<%s> transposing producers...", t.name)
                t2, transOp = createTransposeHelper(t, True)
                self.value_info[t2.name] = t2
                ii = getMaxIndex(t.producers, transOp) + 1
                op2insertIndex.append((transOp, ii))
                for op in t.producers:
                    if op is not transOp:
                        op.replaceOutput(t, t2)

        # insert transpose op to graph
        op2insertIndex = sorted(op2insertIndex, key=lambda k: k[1], reverse=True)
        for op, index in op2insertIndex:
            self.ops.insert(index, op)

    def __str__(self):
        string = str()
        for op in self.ops:
            string += '[OP] ' + str(op) + '\n'
        for t in self.inputs:
            string += '[Inputs] ' + str(t) + '\n'
        for _, t in self.initializer.items():
            string += '[Initializer] ' + str(t) + '\n'
        for _, t in self.value_info.items():
            string += '[Value Info] ' + str(t) + '\n'
        for t in self.outputs:
            string += '[Outputs] ' + str(t) + '\n'
        return string
