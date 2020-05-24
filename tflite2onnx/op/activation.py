import copy
import logging
import tflite
from onnx import helper

from tflite2onnx import tensor
from tflite2onnx.op.operator import Operator

logger = logging.getLogger('tflite2onnx')

FusedActFunc2OpType = {
    tflite.ActivationFunctionType.RELU6: tflite.BuiltinOperator.RELU6,
    tflite.ActivationFunctionType.RELU: tflite.BuiltinOperator.RELU,
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


def handleFusedActivation(master, option, output):
    """Handle FusedActivationFunction for master node.

    For master node such as Conv and FC, there could be FusedActivationFunction.
    If there were, create a activation node `ActOp` and corresponding tensor `actTensor`,
    and insert them into the original graph. E.g. for subgraph `Conv -> convTensor -> OP`
    with `ReLU6`, we generate `Conv -> actTensor -> ActOp(ReLU6) -> convTensor -> OP`.
    """
    logger.debug("Handling FusedActivationFunction for %s", output.name)
    faf = option.FusedActivationFunction()
    if faf is tflite.ActivationFunctionType.NONE:
        output.addProducer(master)
        master.outputs.append(output)
        return

    assert(faf in FusedActFunc2OpType)
    act_type = FusedActFunc2OpType[faf]
    assert(output.status.parsed)

    # create tensor that from Conv/FC to Activation
    input = tensor.Tensor(master.model, master.graph, -1)
    input.name = 'FusedActivation_input_for_%s' % output.name
    input.dtype = output.dtype
    input.layout = copy.deepcopy(output.layout)
    input.shape = copy.deepcopy(output.shape)
    input.setParsed()
    assert(input.name not in tensor.registery)
    tensor.registery[input.name] = input

    # create the activation node, and let master node output to be its'.
    if act_type in [tflite.BuiltinOperator.RELU6, tflite.BuiltinOperator.RELU6]:
        act = ReLU(master.model, master.graph, -1, preset_opcode=act_type)

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

    else:
        raise NotImplementedError("Unsupported fused ActivationFunctionType")

    # attach activation node to output of master node
    it_act = act.inputs[0]
    it_act.addProducer(master)
    master.outputs.append(it_act)
    master.post.append(act)
