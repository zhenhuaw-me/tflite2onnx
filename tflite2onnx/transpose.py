import logging

from tflite2onnx.op import Transpose
from tflite2onnx import layout
from tflite2onnx import tensor

logger = logging.getLogger('tflite2onnx')


def createTransposeHelper(ref, create_as_upstream: bool):
    logger.debug("Creating transpose helper: %s for %s",
                 'upstream' if create_as_upstream else 'downstream', ref)

    if create_as_upstream:
        name = ref.name + '_' + ref.layout.target + '_to_' + ref.layout.source
        newLayout = layout.Layout(ref.layout.target, ref.layout.target)
    else:
        name = ref.name + '_' + ref.layout.source + '_to_' + ref.layout.target
        newLayout = layout.Layout(ref.layout.source, ref.layout.source)
    assert(name not in tensor.registery), "%s should be walked only once" % str(ref)

    # create tensor
    t = tensor.Tensor(ref.model, ref.graph, -1, newLayout, False)
    t.name = name
    t.dtype = ref.dtype
    t.shape = ref.layout.transform(ref.shape)
    t.setParsed()
    assert(t.name not in tensor.registery)
    tensor.registery[t.name] = t

    # create Transpose op
    op = Transpose(ref.model, ref.graph, -1)
    op.name = 'Layout Helper'
    if create_as_upstream:
        input = t
        output = ref
        op.perm = layout.getPerm(ref.layout.target, ref.layout.source)
    else:
        input = ref
        output = t
        op.perm = layout.getPerm(ref.layout.source, ref.layout.target)
    op.inputs.append(input)
    op.outputs.append(output)
    op.setParsed()

    input.addConsumer(op)
    output.addProducer(op)

    return (t, op)
