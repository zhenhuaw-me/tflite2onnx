import logging
import tflite

from tflite2onnx.op.common import Operator

logger = logging.getLogger('tflite2onnx')


class Activation(Operator):
    TypeMapping = {
        tflite.BuiltinOperator.LOGISTIC: 'Sigmoid',
        tflite.BuiltinOperator.PRELU: 'PRelu',
        tflite.BuiltinOperator.RELU6: 'Clip',
        tflite.BuiltinOperator.RELU: 'Relu',
    }

    def __init__(self, TFactory, index, preset_opcode=None):
        super().__init__(TFactory, index)

        # TFLite op code of the activation, e.g. tflite.BuiltinOperator.RELU
        # Used for fused activation, where we cannot parse type from tflite object.
        self.preset_opcode = preset_opcode

        self.setInited()

    @property
    def type(self):
        if self.status.uninitialized:
            return 'Activation'
        else:
            assert(self.tflite or self.preset_opcode), "One of the two must be provided"
            if self.preset_opcode:
                opcode = self.preset_opcode
            else:
                op = self.tflite
                opcode = self.model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
            assert(opcode in self.TypeMapping)
            return self.TypeMapping[opcode]

    def parse(self):
        logger.debug("Parsing %s...", self.type)

        op = self.tflite
        opcode = self.model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
        assert(opcode in self.TypeMapping)

        if opcode == tflite.BuiltinOperator.PRELU:
            assert (op.InputsLength() == 2)
        else:
            assert(op.InputsLength() == 1)
        assert(op.OutputsLength() == 1)

        self.parseInput(0)

        if opcode == tflite.BuiltinOperator.RELU6:
            tmin = self.TFactory.createScalar('float32', 0.0)
            tmin.addConsumer(self)
            self.inputs.append(tmin)
            tmax = self.TFactory.createScalar('float32', 6.0)
            tmax.addConsumer(self)
            self.inputs.append(tmax)

        if opcode == tflite.BuiltinOperator.PRELU:
            # `alpha` should be a learned array with the same shape as `X`
            # But there is no `batch_size` dimension in its shape,
            # which will cause `out of index` exception during axis transform
            # so we expand its dimension by insert 1 to its shape
            alpha = self.parseInput(1)
            alpha.shape.insert(0, 1)

        self.parseOutput(0)

        self.setParsed()

    def propagatableTensors(self):
        return self.inputs + self.outputs

    def transform(self):
        pass


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
    FusedActFunc2OpType = {
        tflite.ActivationFunctionType.RELU6: tflite.BuiltinOperator.RELU6,
        tflite.ActivationFunctionType.RELU: tflite.BuiltinOperator.RELU,
    }

    logger.debug("Handling FusedActivationFunction for %s", master.shorty)
    faf = option.FusedActivationFunction()
    if faf is tflite.ActivationFunctionType.NONE:
        return
    intermediate = master if intermediate is None else intermediate

    assert(faf in FusedActFunc2OpType)
    act_type = FusedActFunc2OpType[faf]
    assert(output.status.parsed)

    # create tensor that from Conv/FC to Activation
    iname = 'TFLITE2ONNX_FAF_%s' % output.name
    input = intermediate.TFactory.getWithRef(output, iname, True)
    input.setParsed()

    intermediate.replaceOutput(output, input)
    input.addProducer(intermediate)

    # create the activation node, and let intermediate node output to be its'.
    if act_type in [tflite.BuiltinOperator.RELU, tflite.BuiltinOperator.RELU6]:
        act = Activation(intermediate.TFactory, -1, preset_opcode=act_type)

        input.addConsumer(act)
        act.inputs.append(input)

        if act_type == tflite.BuiltinOperator.RELU6:
            tmin = intermediate.TFactory.createScalar('float32', 0.0)
            tmin.addConsumer(act)
            act.inputs.append(tmin)
            tmax = intermediate.TFactory.createScalar('float32', 6.0)
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
