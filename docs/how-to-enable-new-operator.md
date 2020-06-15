How to Enable New Operator
==========================

This document will walk you through steps to enable new operators in `tflite2onnx`.
Before going further, make sure that the operator is has not been enabled,
e.g. not included in [Operator Support Status](how-to-enable-new-operator.md).


## Prepare development environment

This is retty simple:

```sh
pip install -r requirements.txt
```


## Generate TensorFlow Lite model

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
eash dimension different extent, such that layout errors can be easily
identified.

Once generated, it's recommended to use visualization tool such as
[Netron](https://github.com/lutzroeder/netron) to verify if the tensors
and operator are what you expected.


## Setup test for the operator

`tflite2onnx` requires test for every operator to ensure that functionality
is not broken across development. The test for operator is also very helpful
when enabling new operators.

Copy the newly generated `model.tflite` to `tflite2onnx` repository, put it
in `${tflite2onnx}/assets/tests`. Naming convesion is `{operator}.{data type}.tflite`,
for example `concat.float32.tflite`. The pattern `{operator}.{data type}`
will be used in our test. Also, `{operator}` doesn't necessarily to be
operator type only, check files in `${tflite2onnx}/assets/tests` for details.

Add the pattern `{operator}.{data type}` into operator test -
`OP_LIST` of function `test_ops()` in `${tflite2onnx}/tests/test_models.py`.
It would help to comment out all other operators and tests when trying around.

Invoke the test `python tests/test_models.py`. You should be able to see errors
like below, which indicates that one operator has not been supported.

```
user@machine [✓] tflite2onnx.git (master*) $ python tests/test_models.py
2020-06-12 23:01:25,856 D [tflite2onnx][convert.py:14] tflite: /Users/user/workspace/onnx/tflite2onnx.git/assets/tests/concat.float32.tflite
2020-06-12 23:01:25,856 D [tflite2onnx][convert.py:15] onnx: concat.float32.onnx
2020-06-12 23:01:25,856 D [tflite2onnx][model.py:21] Parsing the Model...
2020-06-12 23:01:25,857 D [tflite2onnx][graph.py:27] Parsing the Graph...
2020-06-12 23:01:25,857 D [tflite2onnx][graph.py:30] Parsing operator: 0
Traceback (most recent call last):
  File "tests/test_models.py", line 58, in <module>
    test_ops()
  File "tests/test_models.py", line 45, in test_ops
    end2end_test(op)
  File "tests/test_models.py", line 16, in end2end_test
    t2o.convert(tflm_path, onnx_name)
  File "/Users/user/workspace/onnx/tflite2onnx.git/tflite2onnx/convert.py", line 21, in convert
    model.convert()
  File "/Users/user/workspace/onnx/tflite2onnx.git/tflite2onnx/model.py", line 36, in convert
    self.parse()
  File "/Users/user/workspace/onnx/tflite2onnx.git/tflite2onnx/model.py", line 31, in parse
    g.parse()
  File "/Users/user/workspace/onnx/tflite2onnx.git/tflite2onnx/graph.py", line 31, in parse
    op = getOp(self.model, self.graph, i)
  File "/Users/user/workspace/onnx/tflite2onnx.git/tflite2onnx/op/__init__.py", line 32, in getOp
    raise NotImplementedError("Unsupported TFLite OP: {}".format(opcode))
NotImplementedError: Unsupported TFLite OP: 2
```

The `2` of `NotImplementedError: Unsupported TFLite OP: 2` indicates which operator
has not been enabled yet. It is `CONCATENATION` in
[`tflite.BuiltinOperator`](https://github.com/jackwish/tflite/blob/master/tflite/BuiltinOperator.py).

With this, we can really start to write some code.


## Get the operator workflow ready

*TODO: an article introduces how converter works.*

To start with, we add a class to handle converting of this operator.
In this example, we created `tflite.op.Concat` initially as:

```py
class Concat(Operator):
    def __init__(self, model, graph, index):
        super().__init__(model, graph, index)

        self.axis = -1  # operator attribute

        self.setInited()

    @property
    def type(self):
        return 'Concat'

    @property
    def implictLayout(self):
        return False

    def parse(self):
        logger.debug("Parsing %s...", self.type)
        op = self.tflite
        opcode = self.model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
        assert(opcode is tflite.BuiltinOperator.CONCATENATION)

        assert(op.InputsLength() >= 1)
        assert(op.OutputsLength() == 1)

        # TODO: parse tensors

        self.setParsed()

    def convert(self):
        logger.debug("Converting %s...", self.type)

        for t in self.inputs:
            t.convert()
        self.outputs[0].convert()

        inames = [t.name for t in self.inputs]
        onames = [t.name for t in self.outputs]
        logger.debug("Making ONNX...")
        self.onnx = helper.make_node(self.type, inames, onames, axis=self.axis)
```

This can be done by copying an existing similar operator, and make several
modifications.
* In `Operator.__init__()`, it collects the TFLite objects and initializes attributes of the operator. You can take [ONNX Operator Schemas][onnx-op] as reference. When it's done, the object switches to status `INITIALIZED`.
* `Operator.type` is the operator type of ONNX. It's a string which you can find in operator examples in [ONNX Operator Schemas][onnx-op] - usually simply the operator type name, e.g. `Concat` in our example.
* `Operator.implictLayout` describes whether this operator assumes layout of tensors. This is a bit tricky, please look into [Data layout semantic and converting procedure][layout-handling].
* `Operator.parse()` parses the tensors used by the operator, attributes of the operator. Let's left it *to be done* in next section. After finished, set object to status `PARSED`. Mostly, an object should not been parsed multiple times.
* `Operator.convert()` collects the tensors (names actually), attributes that have been parsed, and creates the ONNX operator object, which can be laterly used to create ONNX graph.

Now, let's integrate the operator converter class into framework. This is simple (as we are trying to make it easy to extend :) ).
* Import the class. In `${tflite2onnx}/op/__init__.py`, using this
  ```py
  from tflite2onnx.op.concat import Concat
  ```
* Map the TFLite operator code to the operator converter class in `OP_CONVERTERS`.
  ```py
  OP_CONVERTERS = {
      # other converter class
      tflite.BuiltinOperator.CONCATENATION: Concat,
  }
  ```

That's it! Simple!

Now let's try it. You may see errors like below - take it easy,
as we have not finish our jobs. But we can see that the `Concat`
class is parsing something (nothing so far). That means we have
enabled basic workflow for the operator.

```
wzh@Mac[✗]tflite2onnx.git (master*) $ python tests/test_models.py
2020-06-13 07:57:02,091 D [tflite2onnx][convert.py:14] tflite: /Users/wzh/workspace/onnx/tflite2onnx.git/assets/tests/concat.float32.tflite
2020-06-13 07:57:02,091 D [tflite2onnx][convert.py:15] onnx: concat.float32.onnx
2020-06-13 07:57:02,092 D [tflite2onnx][model.py:21] Parsing the Model...
2020-06-13 07:57:02,092 D [tflite2onnx][graph.py:27] Parsing the Graph...
2020-06-13 07:57:02,092 D [tflite2onnx][graph.py:30] Parsing operator: 0
2020-06-13 07:57:02,092 D [tflite2onnx][concat.py:30] Parsing Concat...
2020-06-13 07:57:02,092 D [tflite2onnx][graph.py:38] Parsing inputs...
2020-06-13 07:57:02,093 D [tflite2onnx][model.py:37] Converting...
2020-06-13 07:57:02,093 D [tflite2onnx][graph.py:69] Converting...
Traceback (most recent call last):
  File "tests/test_models.py", line 58, in <module>
    test_ops()
  File "tests/test_models.py", line 45, in test_ops
    end2end_test(op)
  File "tests/test_models.py", line 16, in end2end_test
    t2o.convert(tflm_path, onnx_name)
  File "/Users/wzh/workspace/onnx/tflite2onnx.git/tflite2onnx/convert.py", line 21, in convert
    model.convert()
  File "/Users/wzh/workspace/onnx/tflite2onnx.git/tflite2onnx/model.py", line 39, in convert
    g.convert()
  File "/Users/wzh/workspace/onnx/tflite2onnx.git/tflite2onnx/graph.py", line 73, in convert
    logger.debug("Graph:\n%s", str(self))
  File "/Users/wzh/workspace/onnx/tflite2onnx.git/tflite2onnx/graph.py", line 151, in __str__
    string += '[Inputs] ' + str(t) + '\n'
  File "/Users/wzh/workspace/onnx/tflite2onnx.git/tflite2onnx/tensor.py", line 116, in __str__
    return '%s: %s -> %s' % (self.str, producer_names, consumer_names)
  File "/Users/wzh/workspace/onnx/tflite2onnx.git/tflite2onnx/tensor.py", line 111, in str
    return '<%s>(%s,%s)' % (self.name, DTYPE_ONNX2NAME[self.dtype], str(self.shape))
KeyError: None
```


## Make the operator converter work

### Understand operator semantic difference

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


### Parse the tensors

Parsing input and output tensors is simple.

In our case, `Concat` may has multiple inputs, so we parse them one by one. For a tensor:
1. Get the tensor index (regarding the TFLite graph).
2. Create the tensor object `tflite2onnx.Tensor` and parse it. Sometimes the tensor has been created already, in which case simply obtain the object.
3. Add it into the graph.

```py
for i in range(op.InputsLength()):
    ii = op.Inputs(i)
    it = tensor.get(self.model, self.graph, ii)
    it.parse()
    it.addConsumer(self)
    self.inputs.append(it)
```

Output is similar.

```py
oi = op.Outputs(0)
ot = tensor.get(self.model, self.graph, oi)
ot.parse()
ot.addConsumer(self)
self.outputs.append(ot)
```

Now, if you invoke the test, you may see that it completes without erros.
You may also catch the tensors and graph log:

```sh
user@Mac[✓]tflite2onnx.git (master*) $ python tests/test_models.py
2020-06-13 09:00:25,018 D [tflite2onnx][convert.py:14] tflite: /Users/user/workspace/onnx/tflite2onnx.git/assets/tests/concat.float32.tflite
2020-06-13 09:00:25,018 D [tflite2onnx][convert.py:15] onnx: concat.float32.onnx
2020-06-13 09:00:25,018 D [tflite2onnx][model.py:21] Parsing the Model...
2020-06-13 09:00:25,018 D [tflite2onnx][graph.py:27] Parsing the Graph...
2020-06-13 09:00:25,018 D [tflite2onnx][graph.py:30] Parsing operator: 0
# pasing the TFLite operator and tensors
2020-06-13 09:00:25,018 D [tflite2onnx][concat.py:30] Parsing Concat...
2020-06-13 09:00:25,019 D [tflite2onnx][tensor.py:87] Parsing a...
2020-06-13 09:00:25,019 D [tflite2onnx][tensor.py:87] Parsing b...
2020-06-13 09:00:25,019 D [tflite2onnx][tensor.py:87] Parsing c...
2020-06-13 09:00:25,019 D [tflite2onnx][tensor.py:87] Parsing Identity...
2020-06-13 09:00:25,019 D [tflite2onnx][graph.py:38] Parsing inputs...
2020-06-13 09:00:25,019 D [tflite2onnx][model.py:37] Converting...
2020-06-13 09:00:25,019 D [tflite2onnx][graph.py:69] Converting...
# What the ONNX graph will have
2020-06-13 09:00:25,020 D [tflite2onnx][graph.py:73] Graph:
[OP] [Unintialized](Concat): ['a', 'b', 'c'] -> ['Identity']
[Inputs] <a>(float32,[1, 1, 2, 3, 1]): [] -> ['[Unintialized](Concat)']
[Inputs] <b>(float32,[1, 1, 2, 3, 2]): [] -> ['[Unintialized](Concat)']
[Inputs] <c>(float32,[1, 1, 2, 3, 3]): [] -> ['[Unintialized](Concat)']
[Value Info] <a>(float32,[1, 1, 2, 3, 1]): [] -> ['[Unintialized](Concat)']
[Value Info] <b>(float32,[1, 1, 2, 3, 2]): [] -> ['[Unintialized](Concat)']
[Value Info] <c>(float32,[1, 1, 2, 3, 3]): [] -> ['[Unintialized](Concat)']
[Value Info] <Identity>(float32,[1, 1, 2, 3, 6]): [] -> ['[Unintialized](Concat)']
[Outputs] <Identity>(float32,[1, 1, 2, 3, 6]): [] -> ['[Unintialized](Concat)']
# creating the ONNX objects
2020-06-13 09:00:25,020 D [tflite2onnx][concat.py:62] Converting Concat...
2020-06-13 09:00:25,020 D [tflite2onnx][tensor.py:100] Converting a...
2020-06-13 09:00:25,021 D [tflite2onnx][tensor.py:100] Converting b...
2020-06-13 09:00:25,021 D [tflite2onnx][tensor.py:100] Converting c...
2020-06-13 09:00:25,021 D [tflite2onnx][tensor.py:100] Converting Identity...
2020-06-13 09:00:25,021 D [tflite2onnx][concat.py:70] Making ONNX...
2020-06-13 09:00:25,022 D [tflite2onnx][graph.py:78] Making ONNX...
2020-06-13 09:00:25,022 D [tflite2onnx][model.py:53] saving model as concat.float32.onnx
2020-06-13 09:00:25,041 I [tflite2onnx][convert.py:23] Converted ONNX model: concat.float32.onnx
```

But, that's not the end necessarily, keep going!


## Parse operator attributes

TFLite model stores *operator option* with dedicated class per operator,
which needs to be handled seperately.

Taking `Concat` example, the options are aviable to obtain after a option
object has *init* from memory. See below.

```py
op_opt = op.BuiltinOptions()
option = tflite.ConcatenationOptions()
option.Init(op_opt.Bytes, op_opt.Pos)
self.axis = option.Axis()
```

Each operator option has a funtion to extract the information, please refer
to the [TFLite parser API][tflite-api].

A TFLite operator option doesn't necessarily have a peer ONNX operator
attribute, vice verse. A TFLite operator option may become ONNX operator input,
or implict ONNX operator semantic. Please do take care consideration for these
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
oi = op.Outputs(0)
ot = tensor.get(self.model, self.graph, oi)
ot.parse()
# ot.addConsumer(self)      # skip this when have FusedActivationFunction()
# self.outputs.append(ot)   # skip this when have FusedActivationFunction()

handleFusedActivation(self, option, ot)
```

Do remeber to initialize ONNX attributes in `Operator.__init__()`.
And do NOT miss any when creating ONNX operator.


## Going further

So far, we have walked the basic routine of enabling a new operator in
`tflite2onnx`. We have not coverted the layout handling in this example.
And, we are not going to have one, because it is sort complicated. If
you met such scenario, you may find a way out, or open discussion. The
important thing is that, this is really the most significant issue for
`tflite2onnx`.

If everthing looks all good. Please update the [Operator Support Status](how-to-enable-new-operator.md), and open pull request.

Thank you for contributing to this project!


[onnx-op]: https://github.com/onnx/onnx/blob/master/docs/Operators.md
[layout-handling]: https://github.com/jackwish/tflite2onnx/issues/2
[tflite-api]: https://jackwish.net/tflite/docs

