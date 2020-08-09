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
        raise NotImplementedError("Method Operator.type() must be overrided!")

    @property
    def implicitLayout(self):
        """Whether the operator assumes implact layout of tensors"""
        raise NotImplementedError("Method %s.implicitLayout() must be overrided!" % self.type)

    def dequantize(self):
        """Dequantize quantized tensors back to float.

        If TFLite model is quantized, we need to dequantize the parsed
        quantized tensors back to float since ONNX has poor quantization
        support. Only tensors of `QLinearConv` and `QLinearMatMul` quantized,
        for which we need to insert `QuantizeLinear` and `DequantizeLinear`
        before and after these two operators.
        """
        raise NotImplementedError("Method %s.dequantize() must be overrided!" % self.type)

    def simpleDequantize(self):
        """Dequantize the tensors only.

        For most of the operators, only need to dequantize the tensors.
        In practice, changing data type for activations and real dequantization
        for initializers (weights).
        """
        for tensor in self.inputs + self.outputs:
            tensor.dequantize()

    def transform(self):
        raise NotImplementedError("Method %s.transform() must be overrided!" % self.type)

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
