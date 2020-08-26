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
    def implicitLayout(self):
        return False

    @property
    def layoutPropagatable(self):
        return True

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

    def transform(self):
        pass

    def convert(self):
        logger.debug("Converting %s...", self.type)

        for t in self.inputs:
            t.convert()
        self.outputs[0].convert()

        inames = [t.name for t in self.inputs]
        onames = [t.name for t in self.outputs]
        self.onnx = helper.make_node(self.type, inames, onames)
        self.setConverted()


def handleFusedActivation(master, option, output, intermediate=None):
    """Handle FusedActivationFunction for master node.

    For master node such as Conv and FC, there could be
    FusedActivationFunction. If there were, create a activation node
    `ActOp` and corresponding tensor `actTensor`, and insert them
    into the original graph. E.g. for subgraph `[Conv] -> <t1>`
    with `ReLU`, we generate `[Conv] -> <t1'> -> [ActOp(ReLU)] -> <t1>`.

    Sometimes, there will be other nodes (quantization node for example)
    inserted between the *master* node and activation node. For such case,
    we cannot attach activation node to master node directly, e.g. the input graph
    will be like `[Conv] -> <t1> -> [Dequantize] -> <t2>`. Therefore, generating
    `[Conv] -> <t1> -> [Dequantize] -> <t2'> -> [ActOp(ReLU)] -> <t2>`.

    So this util generates a pattern `<new tensor> -> [ActOp(ReLU)]` and
    insert to the original graph. In general, we need:
    * `master`: the *mater* node, and usualy which activation attached to.
    * `option`: the option parsed from the original master node.
    * `output`: the tensor that act as output of the whole pattern.
    * `intermediate`: the node that activation attach to, usually same as `master`.
    """
    logger.debug("Handling FusedActivationFunction for %s", master)
    faf = option.FusedActivationFunction()
    if faf is tflite.ActivationFunctionType.NONE:
        return
    intermediate = master if intermediate is None else intermediate

    assert(faf in FusedActFunc2OpType)
    act_type = FusedActFunc2OpType[faf]
    assert(output.status.parsed)

    # create tensor that from Conv/FC to Activation
    input = tensor.Tensor(intermediate.model, intermediate.graph, -1)
    input.name = 'TFLITE2ONNX_FAF_%s' % output.name
    input.dtype = output.dtype
    input.scale = copy.deepcopy(output.scale)
    input.zero_point = copy.deepcopy(output.zero_point)
    input.layout = copy.deepcopy(output.layout)
    input.shape = copy.deepcopy(output.shape)
    input.setParsed()
    assert(input.name not in tensor.registery)
    tensor.registery[input.name] = input

    intermediate.replaceOutput(output, input)
    input.addProducer(intermediate)

    # create the activation node, and let intermediate node output to be its'.
    if act_type in [tflite.BuiltinOperator.RELU, tflite.BuiltinOperator.RELU6]:
        act = ReLU(intermediate.model, intermediate.graph, -1, preset_opcode=act_type)

        input.addConsumer(act)
        act.inputs.append(input)

        if act_type == tflite.BuiltinOperator.RELU6:
            tmin = tensor.createScalar(input, 0)
            tmin.addConsumer(act)
            act.inputs.append(tmin)
            tmax = tensor.createScalar(input, 6)
            tmax.addConsumer(act)
            act.inputs.append(tmax)

        output.replaceProducer(intermediate, act)
        act.outputs.append(output)

        act.setParsed()

        # this is where we need *master* node, all tflite2onnx generated
        # node shall be added as `pre` or `post` of the node that has a TFLite op.
        master.post.append(act)
    else:
        raise NotImplementedError("Unsupported fused ActivationFunctionType")
