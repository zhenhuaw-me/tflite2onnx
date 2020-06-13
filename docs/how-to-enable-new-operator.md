How to Enable New Operator
==========================

This document will walk you through steps to enable new operators in `tflite2onnx`.
Before going further, make sure that the operator is has not been enabled,
e.g. not included in [Operator Support Status](how-to-enable-new-operator.md).


## Prepare development environment

TBD.


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
    def sensitive(self):
        return True

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
* `Operator.sensitive` describes whether this operator is sensitive to layout. This is a bit tricky, please look into [Data layout semantic and converting procedure][layout-handling].
* `Operator.parse()` parses the tensors used by the operator, attributes of the operator. Let's left it *to be done* in next section.
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



[onnx-op]: https://github.com/onnx/onnx/blob/master/docs/Operators.md
[layout-handling]: https://github.com/jackwish/tflite2onnx/issues/2

