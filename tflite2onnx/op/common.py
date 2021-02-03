import logging
import tflite
from onnx import helper

from tflite2onnx.common import T2OBase

logger = logging.getLogger('tflite2onnx')


class Operator(T2OBase):
    TypeMapping = dict()

    def __init__(self, TFactory, index):
        super().__init__(TFactory.model, TFactory.graph, index)
        self.TFactory = TFactory
        self.tflite = self.graph.Operators(index) if index >= 0 else None
        self.inputs = []
        self.outputs = []
        self.pre = []  # ops that before this op which to enable TFLite op
        self.post = []  # ops that after this op which to enable TFLite op
        self.attrs = dict()  # One dict to hold all ONNX operator attributes

    @property
    def type(self):
        raise NotImplementedError("Method Operator.type() must be overrided!")

    def propagatableTensors(self):
        """Get all layout propagable tensors of this operator.

        When we propagate layouts across the graph:
        1. Some operators may stop the propagation
            a) An operator assumes layouts of its tensors, `Conv` for example.
               Such operator needs to define the layouts of its tensors explicitly.
            b) An operator breaks layout semantic, `Reshape` for example.
               Tensors connected to this operator should be propagated.
               And the operator may need special handling regarding layout.
        2. Others may not - propagatable:
            a) An operator that is transparent to layout, such as Add.
               Just propagate the layouts.
            b) Layout can propagate across tensors of an operator, but the operator
               itself has attribution that is sensitive to layout.
               Operator needs special handling after propagation.
        This is defined per operator.

        To handle this, we firstly propagate layouts of tensors across the graph,
        and then update attributes of operators accordingly.
        """
        raise NotImplementedError("Method %s.propagatableTensors() must be overrided!" % self.type)

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

    def parseInput(self, index, layout=None, is_bias=False):
        ii = self.tflite.Inputs(index)
        it = self.TFactory.get(ii, layout, is_bias)
        it.parse()
        it.addConsumer(self)
        self.inputs.append(it)
        return it

    def parseOutput(self, index, layout=None):
        oi = self.tflite.Outputs(index)
        ot = self.TFactory.get(oi, layout)
        ot.parse()
        ot.addProducer(self)
        self.outputs.append(ot)
        return ot

    def replaceInput(self, original, new):
        logger.debug("Replacing %s input %s with %s", self.shorty, original.shorty, new.shorty)
        assert(original in self.inputs)
        for i, item in enumerate(self.inputs):
            if item is original:
                self.inputs[i] = new
                return

    def replaceOutput(self, original, new):
        logger.debug("Replacing %s output %s with %s", self.shorty, original.shorty, new.shorty)
        assert(original in self.outputs)
        for i, item in enumerate(self.outputs):
            if item is original:
                self.outputs[i] = new
                return

    def setParsed(self):
        """Name the operator (if not yet) and change to initialized.

        Assume that the outputs won't change after parsed.
        * If the operator is a helper in TFLITE2ONNX, it should have been named already.
        * If the operator is original in TFLite, using name of its first output tensor.
        """
        self.name = self.outputs[0].name if self.name is None else self.name
        super().setParsed()

    def validate(self):
        assert(len(self.outputs) >= 1), "Operator should produce something"

    def convert(self):
        logger.debug("Converting %s...", self.shorty)
        for t in self.inputs + self.outputs:
            t.convert()
        self.attrs['name'] = self.name
        inames = [t.name for t in self.inputs]
        onames = [t.name for t in self.outputs]
        self.onnx = helper.make_node(self.type, inames, onames, **self.attrs)
        self.setConverted()

    @property
    def shorty(self):
        return '[%s](%s)' % (self.name, self.type)

    def __str__(self):
        inames = str([t.name for t in self.inputs])
        onames = str([t.name for t in self.outputs])
        return '%s attr%s: %s -> %s' % (self.shorty, self.attrs, inames, onames)


class OpFactory:
    """The factory for creating operater converter objects."""

    registry = dict()

    @staticmethod
    def register(converter):
        opcs = converter.TypeMapping.keys()
        for opc in opcs:
            assert(opc not in OpFactory.registry)
            OpFactory.registry[opc] = converter

    def __init__(self, TFactory):
        self.model = TFactory.model
        self.graph = TFactory.graph
        self.TFactory = TFactory

    def create(self, index):
        op = self.graph.Operators(index)
        opcode = self.model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
        if opcode not in OpFactory.registry:
            if opcode in tflite.BUILTIN_OPCODE2NAME:
                name = tflite.opcode2name(opcode)
                raise NotImplementedError("Unsupported TFLite OP: {} {}!".format(opcode, name))
            else:
                raise ValueError("Opcode {} is not a TFLite builtin operator!".format(opcode))

        op_converter = OpFactory.registry[opcode]
        return op_converter(self.TFactory, index)

    @staticmethod
    def dump():
        return "Registered OP converter: %d" % len(OpFactory.registry)

    def __str__(self):
        return OpFactory.dump()
