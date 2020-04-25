import logging
from abc import ABC
from enum import Enum

logger = logging.getLogger('tflite2onnx')


class Status(Enum):
    UNINITIALIZED = 0
    INITIALIZED = 1
    PARSED = 2
    GRAPH_BUILT = 3
    PROPAGATED = 4
    CONVERTED = 5
    INVALID = 10


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

    def buildGraph(self):
        logger.warn("method buildGraph() is not overrided!")

    def setGraphBuilt(self):
        assert(self.status is Status.PARSED)
        self.status = Status.GRAPH_BUILT

    def propagate(self):
        logger.warn("method propagate() is not overrided!")

    def setPropagated(self):
        assert(self.status is Status.GRAPH_BUILT)
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
