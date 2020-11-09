How to Enable New Operator
==========================

This document will walk you through steps to enable new operators in `tflite2onnx`.

It's highly recommended to read the [blog][blog] which introduces the
background of `tflite2onnx`. I am sure it will help when enabling new operators.
Also, make sure that the operator is has not been enabled,
i.e. not included in [Operator Support Status](how-to-enable-new-operator.md).


## Prepare Your Development Environment

This is pretty simple:

```sh
pip install -r requirements.txt
```


## Generate the TensorFlow Lite model

First of all, we need a TensorFlow Lite model (`model.tflite`) to get started.
Currently, to generate a TFLite model, we build a TensorFlow or Keras model,
and convert it into TFLite model.

Below is an example of generating a TFLite model which contains `Concat` operator
only.

```py
import tensorflow as tf

# operator inputs
a = tf.keras.Input(dtype='float32', name='a', shape=(1, 2, 3, 1))
b = tf.keras.Input(dtype='float32', name='b', shape=(1, 2, 3, 2))
c = tf.keras.Input(dtype='float32', name='c', shape=(1, 2, 3, 3))

# operator
concat = tf.keras.layers.Concatenate(axis=-1, name='output')([a, b, c])

# build Keras model
model = tf.keras.Model(inputs=[a, b, c], outputs=[concat])

# convert to TFLite model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# save it
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)
```

Usually, tensor sizes are kept small to generate small model, as some of
them will be hosted in `tflite2onnx` repository. In addition, we gave
each dimension different extent, such that layout errors can be easily
identified.

Once generated, it's recommended to use visualization tool such as
[Netron](https://github.com/lutzroeder/netron) to verify if the tensors
and operator are what you expect.


## Setup Test for the Operator

`tflite2onnx` requires test for every operator to ensure that functionality
is not broken across development. The test for the operator is also very
helpful when enabling new operators.

Copy the newly generated `model.tflite` to `tflite2onnx` repository, put it
in `${tflite2onnx}/assets/tests`. Naming convesion is `{operator}.{data type}.tflite`,
for example `concat.float32.tflite`. The pattern `{operator}.{data type}`
will be used in our test. Also, `{operator}` doesn't necessarily to be
operator type only, check files in `${tflite2onnx}/assets/tests` for details.

Add the pattern `{operator}.{data type}` into operator test in
`${tflite2onnx}/tests/test_ops.py`, depending the data layout attribution of
the operator (if you don't know which sub test shall the op goes, check the [blog][blog]).
It would help to comment out all other operators and tests when trying around.

Invoke the test `python tests/test_ops.py`. You should be able to see errors
like below, which indicates that one operator has not been supported.

```
wzh@Mac[✓]tflite2onnx.git (master*) $ python tests/test_ops.py
2020-11-09 20:51:00,439 D [tflite2onnx][convert.py:37] tflite: /Users/wzh/workspace/onnx/tflite2onnx.git/assets/tests/concat.float32.tflite
2020-11-09 20:51:00,439 D [tflite2onnx][convert.py:38] onnx: concat.float32.onnx
2020-11-09 20:51:00,439 D [tflite2onnx][model.py:21] Parsing the Model...
2020-11-09 20:51:00,439 D [tflite2onnx][graph.py:58] Parsing the Graph...
2020-11-09 20:51:00,439 D [tflite2onnx][graph.py:61] Parsing operator: 0
Traceback (most recent call last):
  File "tests/test_ops.py", line 85, in <module>
    test_ops_post_propagation()
  File "tests/test_ops.py", line 64, in test_ops_post_propagation
    end2end_test(op, 'NHWC')
  File "tests/test_ops.py", line 16, in end2end_test
    t2o.convert(tflm_path, onnx_name)
  File "/Users/wzh/workspace/onnx/tflite2onnx.git/tflite2onnx/convert.py", line 44, in convert
    model.convert(explicit_layouts)
  File "/Users/wzh/workspace/onnx/tflite2onnx.git/tflite2onnx/model.py", line 39, in convert
    self.parse()
  File "/Users/wzh/workspace/onnx/tflite2onnx.git/tflite2onnx/model.py", line 31, in parse
    g.parse()
  File "/Users/wzh/workspace/onnx/tflite2onnx.git/tflite2onnx/graph.py", line 62, in parse
    op = self.OPCFactory.create(i)
  File "/Users/wzh/workspace/onnx/tflite2onnx.git/tflite2onnx/op/common.py", line 151, in create
    raise NotImplementedError("Unsupported TFLite OP: {}".format(opcode))
NotImplementedError: Unsupported TFLite OP: 2
```

The `2` of `NotImplementedError: Unsupported TFLite OP: 2` indicates which operator
has not been enabled yet. It is `CONCATENATION` in
[`tflite.BuiltinOperator`](https://github.com/jackwish/tflite/blob/master/tflite/BuiltinOperator.py).

With this, we can really start to write some code.


## Get the Workflow of the Operator Ready

To start with, we add a operator converter class to handle
converting of this operator.
In this example, we created `tflite.op.Concat` initially as:

```py
class Concat(Operator):
    TypeMapping = {
        tflite.BuiltinOperator.CONCATENATION: 'Concat',
    }

    def __init__(self, TFactory, index):
        super().__init__(TFactory, index)

        self.attrs['axis'] = -1 # operator attribute

        self.setInited()

    @property
    def type(self):
        return 'Concat'

    def parse(self):
        logger.debug("Parsing %s...", self.type)
        op = self.tflite
        opcode = self.model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
        assert(opcode is tflite.BuiltinOperator.CONCATENATION)

        assert(op.InputsLength() >= 1)
        assert(op.OutputsLength() == 1)

        # TODO: parse tensors

        self.setParsed()

    def propagatableTensors(self):
        return list()

    def transform(self):
        # TODO: handle layout transform
        pass
```

This can be done by copying an existing similar operator, and make several
modifications.
* `Operator.TypeMapping` maps TFLite operator type to ONNX operator type. Different TFLite operator may map to same ONNX operator type. An operator converter may be able to handle many TFLite operators.
* `Operator.__init__()` collects the TFLite objects and initializes attributes of the operator. You can take [ONNX Operator Schemas][onnx-op] as reference. When it's done, the object switches to status `INITIALIZED`.
* `Operator.type` is the operator type of ONNX. It's a string which you can find in operator examples in [ONNX Operator Schemas][onnx-op] - usually simply the operator type name, e.g. `Concat` in our example. The type may be mapped from the TFLite operator type via `Operator.TypeMapping` or other information sometimes depending on the implementation.
* `Operator.parse()` parses the tensors used by the operator, attributes of the operator. Let's left it *to be done* in next section. After finished, set object to status `PARSED`. Mostly, an object should not been parsed multiple times.
* `Operator.propagatableTensors()` describes which tensors of this operator are layout propagatable. This is a bit tricky, please look into [Data layout semantic and converting procedure][layout-handling].
* `Operator.transform()` transforms operator attributes that are sensitive to layout. This is sort tricky which requires serious consideration. Leave it empty currently.

Now, let's integrate the operator converter class into framework.
This is simple (as we are trying to make it easy to extend :) ).
Import and register the operator converter class. In `${tflite2onnx}/op/__init__.py`, add code below.

```py
from tflite2onnx.op.concat import Concat
# ...

OpFactory.register(Concat)
```

That's it! Simple!

Now let's try it. You may see errors like below - take it easy,
as we have not finish our jobs. But we can see that the `Concat`
class is parsing something (nothing so far). That means we have
enabled basic workflow for the operator.

```
2020-11-09 21:04:45,344 D [tflite2onnx][convert.py:37] tflite: /Users/wzh/workspace/onnx/tflite2onnx.git/assets/tests/concat.float32.tflite
2020-11-09 21:04:45,345 D [tflite2onnx][convert.py:38] onnx: concat.float32.onnx
2020-11-09 21:04:45,345 D [tflite2onnx][model.py:21] Parsing the Model...
2020-11-09 21:04:45,345 D [tflite2onnx][graph.py:58] Parsing the Graph...
2020-11-09 21:04:45,345 D [tflite2onnx][graph.py:61] Parsing operator: 0
2020-11-09 21:04:45,345 D [tflite2onnx][concat.py:27] Parsing [None](Concat)...
Traceback (most recent call last):
  File "tests/test_ops.py", line 85, in <module>
    test_ops_post_propagation()
  File "tests/test_ops.py", line 64, in test_ops_post_propagation
    end2end_test(op, 'NHWC')
  File "tests/test_ops.py", line 16, in end2end_test
    t2o.convert(tflm_path, onnx_name)
  File "/Users/wzh/workspace/onnx/tflite2onnx.git/tflite2onnx/convert.py", line 44, in convert
    model.convert(explicit_layouts)
  File "/Users/wzh/workspace/onnx/tflite2onnx.git/tflite2onnx/model.py", line 39, in convert
    self.parse()
  File "/Users/wzh/workspace/onnx/tflite2onnx.git/tflite2onnx/model.py", line 31, in parse
    g.parse()
  File "/Users/wzh/workspace/onnx/tflite2onnx.git/tflite2onnx/graph.py", line 63, in parse
    op.parse()
  File "/Users/wzh/workspace/onnx/tflite2onnx.git/tflite2onnx/op/concat.py", line 47, in parse
    self.setParsed()
  File "/Users/wzh/workspace/onnx/tflite2onnx.git/tflite2onnx/op/common.py", line 104, in setParsed
    self.name = self.outputs[0].name if self.name is None else self.name
IndexError: list index out of range
```


## Make the Operator Converter Work

### Understand Operator Semantic Divergence

TFLite and ONNX operator semantic are sometimes different. Make sure to review
[TFLite API documentation][tflite-api] for operator option,
and [ONNX documents][onnx-op] for operator attributes. To be noted, some operator
option or attribute of one, could be described by input tensor in another.

For this `Concat` example, it accepts several inputs and generate one output in both
TFLite and [ONNX](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Concat).
Unfortuanately, TFLite doesn't provide rich documents about operators, we may check
[Compatible operations of TensorFlow and TFLite](https://www.tensorflow.org/lite/guide/ops_compatibility#compatible_operations)
and sometimes even the [source code](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/kernels/internal/reference/concatenation.h).

For options or attributes, we can check the
[*OperatorOption* of TFLite](https://jackwish.net/tflite/docs/ConcatenationOptions.m.html#tflite.ConcatenationOptions.ConcatenationOptions).
In our `Concat` example, it has two:
* `Axis` indicates concatenating on which dimension. This attribute is sensitive to how we handle layout issue. For example, if TFLite concatenates on axis `-1` and has a `NHWC` data layout - which means it's concatenating on `C` dimension. While ONNX uses layout `NCHW`, ONNX version needs to concatenates on axis `1` for it's dimension `C` in ONNX. This is needed when `Concat` feeds to a `Conv`. The interesting part is, if the model contains no `Conv`, for example has only one `Concat` in our case, we'd better keep it unchanged. We will discuss this more in dedicated document.
* `FusedActivationFunction` describes which [activation function](https://jackwish.net/tflite/docs/ActivationFunctionType.m.html) has been fused into this operator. This is commen in operators like `Conv` and `FullyConnected`.


### Parse the Tensors

Parsing input and output tensors is simple, and we have provided
well wrapped helpers to make this easy.

In our case, `Concat` may has multiple inputs and one output, so just

```py
for i in range(op.InputsLength()):
    self.parseInput(i)

self.parseOutput(0)
```

Now, if you invoke the test, you may see that it completes without erros.
You may also catch the tensors and graph log (well, we have many debug
log to make investigation easier):

```sh
wzh@Mac[✗]tflite2onnx.git (master*) $ python tests/test_ops.py
2020-11-09 21:11:24,589 D [tflite2onnx][convert.py:37] tflite: /Users/wzh/workspace/onnx/tflite2onnx.git/assets/tests/concat.float32.tflite
2020-11-09 21:11:24,589 D [tflite2onnx][convert.py:38] onnx: concat.float32.onnx
2020-11-09 21:11:24,589 D [tflite2onnx][model.py:21] Parsing the Model...
2020-11-09 21:11:24,589 D [tflite2onnx][graph.py:58] Parsing the Graph...
2020-11-09 21:11:24,589 D [tflite2onnx][graph.py:61] Parsing operator: 0
2020-11-09 21:11:24,590 D [tflite2onnx][concat.py:27] Parsing [None](Concat)...
2020-11-09 21:11:24,590 D [tflite2onnx][tensor.py:103] Parsing a...
2020-11-09 21:11:24,590 D [tflite2onnx][tensor.py:103] Parsing b...
2020-11-09 21:11:24,591 D [tflite2onnx][tensor.py:103] Parsing c...
2020-11-09 21:11:24,591 D [tflite2onnx][tensor.py:103] Parsing Identity...
2020-11-09 21:11:24,591 D [tflite2onnx][model.py:40] Converting...
2020-11-09 21:11:24,591 D [tflite2onnx][graph.py:91] Converting...
2020-11-09 21:11:24,592 D [tflite2onnx][graph.py:93] Handling data layout...
2020-11-09 21:11:24,592 D [tflite2onnx][graph.py:130] Propragating layout across graph...
2020-11-09 21:11:24,592 D [tflite2onnx][graph.py:141] Propagation: 4 tensors in total, 0 to walk, 4 at wild
2020-11-09 21:11:24,592 D [tflite2onnx][graph.py:164] Propagation: wild tensors 4, ignored tensors 0
2020-11-09 21:11:24,592 D [tflite2onnx][graph.py:104] Translating quantization semantic...
2020-11-09 21:11:24,592 D [tflite2onnx][graph.py:112] Graph:
[OP] [Identity](Concat) attr{'axis': -1}: ['a', 'b', 'c'] -> ['Identity']
[Input] <a>(float32,[1, 1, 2, 3, 1]): {[]} -> {['[Identity](Concat)']}
[Input] <b>(float32,[1, 1, 2, 3, 2]): {[]} -> {['[Identity](Concat)']}
[Input] <c>(float32,[1, 1, 2, 3, 3]): {[]} -> {['[Identity](Concat)']}
[Output] <Identity>(float32,[1, 1, 2, 3, 6]): {['[Identity](Concat)']} -> {[]}
[Value Info] <a>(float32,[1, 1, 2, 3, 1]): {[]} -> {['[Identity](Concat)']}
[Value Info] <Identity>(float32,[1, 1, 2, 3, 6]): {['[Identity](Concat)']} -> {[]}
[Value Info] <b>(float32,[1, 1, 2, 3, 2]): {[]} -> {['[Identity](Concat)']}
[Value Info] <c>(float32,[1, 1, 2, 3, 3]): {[]} -> {['[Identity](Concat)']}

2020-11-09 21:11:24,592 D [tflite2onnx][common.py:111] Converting [Identity](Concat)...
2020-11-09 21:11:24,592 D [tflite2onnx][tensor.py:142] Converting <a>(float32,[1, 1, 2, 3, 1])...
2020-11-09 21:11:24,593 D [tflite2onnx][tensor.py:142] Converting <b>(float32,[1, 1, 2, 3, 2])...
2020-11-09 21:11:24,593 D [tflite2onnx][tensor.py:142] Converting <c>(float32,[1, 1, 2, 3, 3])...
2020-11-09 21:11:24,593 D [tflite2onnx][tensor.py:142] Converting <Identity>(float32,[1, 1, 2, 3, 6])...
2020-11-09 21:11:24,593 D [tflite2onnx][graph.py:118] Making ONNX...
2020-11-09 21:11:24,594 D [tflite2onnx][model.py:56] saving model as concat.float32.onnx
2020-11-09 21:11:24,614 I [tflite2onnx][convert.py:46] Converted ONNX model: concat.float32.onnx
2020-11-09 21:11:24,622 D [shrub][onnx.py:55] running concat.float32.onnx
2020-11-09 21:11:24,709 D [shrub][onnx.py:22]  parsing concat.float32.onnx
2020-11-09 21:11:25,384 D [tensorflow][tpu_cluster_resolver.py:34] Falling back to TensorFlow client; we recommended you install the Cloud TPU client directly with pip install cloud-tpu-client.
2020-11-09 21:11:26,565 D [shrub][tflite.py:98] running /Users/wzh/workspace/onnx/tflite2onnx.git/assets/tests/concat.float32.tflite
2020-11-09 21:11:26,568 D [shrub][tflite.py:104] Inputs: [{'name': 'a', 'index': 0, 'shape': array([1, 1, 2, 3, 1], dtype=int32), 'shape_signature': array([1, 1, 2, 3, 1], dtype=int32), 'dtype': <class 'numpy.float32'>, 'quantization': (0.0, 0), 'quantization_parameters': {'scales': array([], dtype=float32), 'zero_points': array([], dtype=int32), 'quantized_dimension': 0}, 'sparsity_parameters': {}}, {'name': 'b', 'index': 1, 'shape': array([1, 1, 2, 3, 2], dtype=int32), 'shape_signature': array([1, 1, 2, 3, 2], dtype=int32), 'dtype': <class 'numpy.float32'>, 'quantization': (0.0, 0), 'quantization_parameters': {'scales': array([], dtype=float32), 'zero_points': array([], dtype=int32), 'quantized_dimension': 0}, 'sparsity_parameters': {}}, {'name': 'c', 'index': 2, 'shape': array([1, 1, 2, 3, 3], dtype=int32), 'shape_signature': array([1, 1, 2, 3, 3], dtype=int32), 'dtype': <class 'numpy.float32'>, 'quantization': (0.0, 0), 'quantization_parameters': {'scales': array([], dtype=float32), 'zero_points': array([], dtype=int32), 'quantized_dimension': 0}, 'sparsity_parameters': {}}]
2020-11-09 21:11:26,569 D [shrub][tflite.py:105] Outputs: [{'name': 'Identity', 'index': 3, 'shape': array([1, 1, 2, 3, 6], dtype=int32), 'shape_signature': array([1, 1, 2, 3, 6], dtype=int32), 'dtype': <class 'numpy.float32'>, 'quantization': (0.0, 0), 'quantization_parameters': {'scales': array([], dtype=float32), 'zero_points': array([], dtyp
```

But, that's not the end necessarily, keep going!


## Parse Operator Attributes

TFLite model stores *operator option* with dedicated class per operator,
which needs to be handled seperately.

Taking `Concat` example, the options are aviable to obtain after a option
object has *init* from memory. See below.

```py
op_opt = op.BuiltinOptions()
option = tflite.ConcatenationOptions()
option.Init(op_opt.Bytes, op_opt.Pos)
self.attrs['axis'] = option.Axis()
```

Each operator option has a funtion to extract the information, please refer
to the [TFLite parser API][tflite-api]. And all ONNX attributes are collected
in `Operator.attrs`.

A TFLite operator option doesn't necessarily have a peer ONNX operator
attribute, vice verse. A TFLite operator option may become ONNX operator input,
or implicit ONNX operator semantic. Please do take care consideration for these
functionalities. If you are not sure, take existing operator converter as
reference, or open issue to ask.

Among all the options, *fused activation function* is one special, for which
we need to add one more ONNX operator to the graph. But, don't worry, it can be
handled by simply calling `handleFusedActivation(self, option, ot)`, if that
operator has a `FusedActivationFunction()`
([`Concat` example](https://jackwish.net/tflite/docs/ConcatenationOptions.m.html#tflite.ConcatenationOptions.ConcatenationOptions.FusedActivationFunction))
method of its option class. If that is the case, please don't add output tensor
of the operator directly, but do something like below.

```py
handleFusedActivation(self, option, ot)
```

Do remeber to initialize ONNX attributes in `Operator.__init__()`.
And do NOT miss any when creating ONNX operator.


## Handling Data Layout Issue

Given that the data layout of TFLite models and ONNX models are NHWC
and NCHW respective, some additional efforts are needed when enabling
operators. If you have not read the [blog][blog] or the [layout handling
story][layout-handling], not it's the time.

`Operator.propagatableTensors()` describes which tensors of this operator
are layout propagatable. For most case like `Concat`, all tensors are
propagatable, so we can write it like this.

```py
def propagatableTensors(self):
    return self.inputs + self.outputs
```

But for operators like `Conv`, none of it's tensors is propagatable.
Be carefull for this part as it may require significant effort to debug
if it's not correctly coded at the begining.

`Operator.transform()` transforms operator attributes that are sensitive
to layout. For most case, this function can be left as empty. But, just like
`Operator.propagatableTensors()`, we need to check what should be done.

For `Concat`, which requires attribute transform, we get the `layout`
description (source layout and target layout) from output, and transform
attribute `axis` accordingly.

```py
def transform(self):
    logger.debug("Transforming %s...", self.shorty)
    layout = self.outputs[0].layout
    if layout is not None:
        axis = self.attrs['axis']
        axis = axis if axis >= 0 else (axis + len(layout.perm))
        self.attrs['axis'] = layout.perm.index(axis)
```

To be noted, the layout issue is case by case, this document only shows
`Concat` as an example. You may find other operator converter classes
as a hint for the operator you are trying to enable.
If you have any question, just open issue to discuss.


## Going Further

Congratulation! You have basically finished the implementation of
a new operator. If everthing looks good, please update the
[Operator Support Status](how-to-enable-new-operator.md), and open
pull request. Let your work empower the community.

Thank you for your contribution!

Cheers!


[onnx-op]: https://github.com/onnx/onnx/blob/master/docs/Operators.md
[layout-handling]: https://github.com/jackwish/tflite2onnx/issues/2
[tflite-api]: https://jackwish.net/tflite/docs
[blog]: https://jackwish.net/2020/Convert-TensorFlow-Lite-models-to-ONNX.html
