import logging
import tflite
from onnx import helper

from tflite2onnx import tensor
from tflite2onnx.op.operator import Operator

logger = logging.getLogger('tflite2onnx')

FusedActFunc2OpType = {
    tflite.ActivationFunctionType.RELU6: tflite.BuiltinOperator.RELU6,
    tflite.BuiltinOperator.RELU: tflite.BuiltinOperator.RELU,
}

OpTypeMapping = {
    tflite.BuiltinOperator.RELU: 'Relu',
    tflite.BuiltinOperator.RELU6: 'Clip',
}


class ReLU(Operator):
    def __init__(self, model, graph, index, preset_opcode=None):
        super().__init__(model, graph, index)
        self.setInited()
        # TFLite op code of the activation, e.g. tflite.BuiltinOperator.RELU
        # Used for fused activation, where we cannot parse type from tflite object.
        self.preset_opcode = preset_opcode

    @property
    def type(self):
        if self.status.uninitialized:
            return 'Relu-family'
        else:
            assert(self.tflite is not self.preset_opcode)
            if self.preset_opcode is not None:
                opcode = self.preset_opcode
            else:
                op = self.tflite
                opcode = self.model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
            assert(opcode in OpTypeMapping)
            return OpTypeMapping[opcode]

    @property
    def sensitive(self):
        return False

    def parse(self):
        logger.debug("Parsing %s...", self.type)

        op = self.tflite
        opcode = self.model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
        assert(opcode in OpTypeMapping)

        assert(op.InputsLength() == 1)
        assert(op.OutputsLength() == 1)

        ii = op.Inputs(0)
        it = tensor.get(self.model, self.graph, ii)
        it.parse()
        it.addConsumer(self)
        self.inputs.append(it)

        if opcode == tflite.BuiltinOperator.RELU6:
            tmin = tensor.createScalar(it, 0)
            tmin.addConsumer(self)
            self.inputs.append(tmin)
            tmax = tensor.createScalar(it, 6)
            tmax.addConsumer(self)
            self.inputs.append(tmax)

        oi = op.Outputs(0)
        ot = tensor.get(self.model, self.graph, oi)
        ot.parse()
        ot.addProducer(self)
        self.outputs.append(ot)

        self.setParsed()

    def convert(self):
        logger.debug("Converting %s...", self.type)

        for t in self.inputs:
            t.convert()
        self.outputs[0].convert()

        inames = [t.name for t in self.inputs]
        onames = [t.name for t in self.outputs]
        self.onnx = helper.make_node(self.type, inames, onames)
        self.setConverted()


def createFusedActivation(model, graph, actf, output):
    assert(actf in FusedActFunc2OpType)
    act_type = FusedActFunc2OpType[actf]
    assert(output.status.parsed)
    act = ReLU(model, graph, -1, preset_opcode=act_type)

    # create tensor that from Conv/FC to Activation
    input = tensor.Tensor(model, graph, -1, layout=output.layout)
    input.name = 'input_of_%s_to_%s' % (OpTypeMapping[act_type], output.name)
    input.dtype = output.dtype
    input.shape = output.shape
    input.setParsed()
    assert(input.name not in tensor.registery)
    tensor.registery[input.name] = input

    input.addConsumer(act)
    act.inputs.append(input)

    if act_type == tflite.BuiltinOperator.RELU6:
        tmin = tensor.createScalar(input, 0)
        tmin.addConsumer(act)
        act.inputs.append(tmin)
        tmax = tensor.createScalar(input, 6)
        tmax.addConsumer(act)
        act.inputs.append(tmax)

    output.addProducer(act)
    act.outputs.append(output)

    act.setParsed()

    return act
