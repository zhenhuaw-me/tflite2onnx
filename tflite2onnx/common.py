import logging
from abc import ABC
from enum import Enum

logger = logging.getLogger('tflite2onnx')


class Status(Enum):
    # Before `__init__()` finishes.
    UNINITIALIZED = 0

    # Basic TensorFlow Lite objects registed, Class-wise member allocated.
    INITIALIZED = 1

    # Objects and any members have been parsed from TFLite model.
    PARSED = 2

    # Everything that needs done in graph walking has been done.
    PROPAGATED = 3

    # ONNX object has been created.
    CONVERTED = 4

    # Reserved.
    INVALID = 10

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
        assert(self.status is Status.UNINITIALIZED)
        self.status = Status.INITIALIZED

    def parse(self):
        logger.warn("method parse() is not overrided!")

    def setParsed(self):
        assert(self.status is Status.INITIALIZED)
        self.status = Status.PARSED

    def propagate(self):
        logger.warn("method propagate() is not overrided!")

    def setPropagated(self):
        assert(self.status is Status.PARSED)
        self.status = Status.PROPAGATED

    def convert(self):
        logger.warn("method convert() is not overrided!")

    def setConverted(self):
        assert(self.status is Status.PROPAGATED)
        self.status = Status.CONVERTED

    def setInvalid(self):
        self.status = Status.INVALID

    def __str__(self):
        return self.onnx.__str__()
