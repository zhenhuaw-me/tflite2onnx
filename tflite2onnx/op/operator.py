import logging
from onnx import helper

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
        self.attrs = dict()  # One dict to hold all ONNX operator attributes

    @property
    def type(self):
        raise NotImplementedError("Method Operator.type() must be overrided!")

    @property
    def layoutPropagatable(self):
        """Should the layout be propagated across this operator?

        When we propagate layouts across the graph:
        1. [False] Some operators may stop the propagation
            a) An operator assumes layouts of its tensors, `Conv` for example.
               Such operator needs to define the layouts of its tensors explicitly.
            b) An operator breaks layout semantic, `Reshape` for example.
               Tensors connected to this operator should be propagated.
               And the operator may need special handling regarding layout.
        2. [True] Others may not - propagatable:
            a) An operator that is transparent to layout, such as Add.
               Just propagate the layouts.
            b) Layout can propagate across tensors of an operator, but the operator
               itself has attribution that is sensitive to layout.
               Operator needs special handling after propagation.
        This is defined per operator.
        """
        raise NotImplementedError("Method %s.layoutPropagatable() must be overrided!" % self.type)

    def transform(self):
        """Transform the operator attributions w.r.t. propagated layouts.

        The attributions could be a tensor that describing layout related things.
        Operators that defined as 1.a, 1.b and 2.b in `layoutPropagatable()`
        are such cases. But not all of them need special treatment.
        For example, `Conv` doesn't need additional processing after propagation.

        This must be called after the layouts have been propagated across graph.
        """
        raise NotImplementedError("Method %s.transform() must be overrided!" % self.type)

    @property
    def str(self):
        return '[' + self.name + '] (' + self.type + ')'

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

    def validate(self):
        assert(len(self.outputs) >= 1), "Operator should produce something"

    def __str__(self):
        inames = str([t.name for t in self.inputs])
        onames = str([t.name for t in self.outputs])
        return self.str + ': ' + inames + ' -> ' + onames

    def convert(self):
        logger.debug("converting %s...", self.type)
        for t in self.inputs + self.outputs:
            t.convert()
        inames = [t.name for t in self.inputs]
        onames = [t.name for t in self.outputs]
        self.onnx = helper.make_node(self.type, inames, onames, **self.attrs)
        self.setConverted()
