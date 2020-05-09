from .common import logger
from .op.transpose import Transpose
from . import layout
from . import tensor


def createTransposeHelper(input, output, upstream):
    logger.debug("Creating layout helper for <%s> -> <%s>", input.name, output.name)
    op = Transpose(input.model, input.graph, -1)
    op.name = 'Layout Helper'
    op.inputs.append(input)
    op.outputs.append(output)
    if upstream:
        l = output.layout
        op.perm = layout.getPerm(l.target, l.source)
    else:
        l = input.layout
        op.perm = layout.getPerm(l.source, l.target)
    op.setParsed()

    input.addConsumer(op)
    output.addProducer(op)

    return op


def createTransposeTensor(ref, upstream):
    # upstream: whether the transpose tensor is upstream of the ref tensor
    assert(ref.name in tensor.registery)
    if upstream:
        l = layout.Layout(ref.layout.target, ref.layout.source)
        thisLayout = layout.Layout(ref.layout.target, ref.layout.target)
    else:
        l = layout.Layout(ref.layout.source, ref.layout.target)
        thisLayout = layout.Layout(ref.layout.source, ref.layout.source)

    name = ref.name + l.nameSuffix
    if name in tensor.registery:
        assert(ref.layout.match)
        return tensor.registery[name]

    assert(ref.onnx is None)
    t = tensor.Tensor(ref.model, ref.graph, -1, thisLayout, False)
    t.name = name
    t.dtype = ref.dtype
    t.shape = ref.layout.transform(ref.shape)
    t.setParsed()
    # ref.layout.markDone() # FIXME
    tensor.registery[name] = t
    return t


