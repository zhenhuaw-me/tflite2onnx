from abc import ABC
from enum import Enum


class Status(Enum):
    # Before `__init__()` finishes.
    UNINITIALIZED = 0

    # Basic TensorFlow Lite objects registed, Class-wise member allocated.
    INITIALIZED = 1

    # Objects and any members have been parsed from TFLite model.
    PARSED = 2

    # ONNX object has been created.
    CONVERTED = 3

    # Reserved.
    INVALID = 10

    @property
    def uninitialized(self):
        return self == self.UNINITIALIZED

    @property
    def initialized(self):
        return self == self.INITIALIZED

    @property
    def parsed(self):
        return self == self.PARSED

    @property
    def converted(self):
        return self == self.CONVERTED


class LayoutApproach(Enum):
    DEFAULT = 1
    TRANSPOSE = 1
    PROPAGATION = 2


class T2OBase(ABC):
    """Holding objects of TFLite and ONNX"""
    def __init__(self, model=None, graph=None, index=None):
        # Overall fields
        self.status = Status.UNINITIALIZED

        # TFLite objects
        self.name = 'Unintialized'
        self.model = model
        self.graph = graph
        self.index = index  # index of tensor or op
        self.tflite = None

        # ONNX object
        self.onnx = None

    def setInited(self):
        assert(self.status.uninitialized)
        self.status = Status.INITIALIZED

    def parse(self):
        raise NotImplementedError("method parse() is not overrided!")

    def setParsed(self):
        assert(self.status.initialized)
        self.status = Status.PARSED

    def convert(self):
        raise NotImplementedError("method convert() is not overrided!")

    def setConverted(self):
        assert(self.status.parsed)
        self.status = Status.CONVERTED

    def setInvalid(self):
        self.status = Status.INVALID

    def __str__(self):
        return self.onnx.__str__()
