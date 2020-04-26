from ..common import T2OBase


class Operator(T2OBase):
    def __init__(self, model, graph, index):
        super().__init__(model, graph, index)
        self.tflite = graph.Operators(index)
        self.inputs = []
        self.outputs = []
        self.weights = []  # there could be overlap between inputs and weights

    @property
    def type(self):
        raise NotImplementedError

    @property
    def sensitive(self):
        raise NotImplementedError
