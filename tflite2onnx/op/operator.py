import logging

from tflite2onnx.common import T2OBase

logger = logging.getLogger('tflite2onnx')


class Operator(T2OBase):
    def __init__(self, model, graph, index):
        super().__init__(model, graph, index)
        self.tflite = graph.Operators(index) if index >= 0 else None
        self.inputs = []
        self.outputs = []
        self.pre = []  # ops that before this op which to enable TFLite op
        self.post = []  # ops that after this op which to enable TFLite op

    @property
    def type(self):
        raise NotImplementedError

    @property
    def sensitive(self):
        raise NotImplementedError

    @property
    def str(self):
        return '[' + self.name + ']' + '(' + str(self.type) + ')'

    def replaceInput(self, original, new):
        logger.debug("Replacing [%s] input <%s> with <%s>", self.type, original.name, new.name)
        for index, item in enumerate(self.inputs):
            if item is original:
                self.inputs[index] = new
                return
        assert(False)

    def replaceOutput(self, original, new):
        logger.debug("Replacing [%s] output <%s> with <%s>", self.type, original.name, new.name)
        for index, item in enumerate(self.outputs):
            if item is original:
                self.outputs[index] = new
                return
        assert(False)

    def __str__(self):
        inames = str([t.name for t in self.inputs])
        onames = str([t.name for t in self.outputs])
        return self.str + ': ' + inames + ' -> ' + onames

    def _convertTensors(self):
        for t in self.inputs:
            t.convert()
        for t in self.outputs:
            t.convert()
