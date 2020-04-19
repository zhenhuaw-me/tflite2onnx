from ..common import T2OBase


class Operator(T2OBase):
    def __init__(self):
        T2OBase.__init__(self)
        self.type = None
        self.inputs = []
        self.outputs = []
        self.weights = []  # there could be overlap between inputs and weights
