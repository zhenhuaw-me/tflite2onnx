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


class T2OBase(ABC):
    """Holding objects of TFLite and ONNX"""
    def __init__(self, model=None, graph=None, index=None):
        # Overall fields
        self.status = Status.UNINITIALIZED
        self.name = None

        # TFLite objects
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
        raise NotImplementedError("method parse() should be overrided!")

    def setParsed(self):
        assert(self.status.initialized)
        self.status = Status.PARSED

    def validate(self):
        raise NotImplementedError("method validate() should be overrided!")

    def convert(self):
        raise NotImplementedError("method convert() should be overrided!")

    def setConverted(self):
        assert(self.status.parsed)
        self.status = Status.CONVERTED

    def setInvalid(self):
        self.status = Status.INVALID

    @property
    def shorty(self):
        """A short readable description for the class/object.

        This aims to be different from `__str__` which is exepcted to be
        long description on this package.
        """
        raise NotImplementedError("method shorty() should be overrided!")

    def __str__(self):
        """A readable description for the class/object."""
        raise NotImplementedError("method __str__() should be overrided!")
