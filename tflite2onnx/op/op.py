from ..common import BaseABC


class Operator(BaseABC):
    def __init__(self):
        BaseABC.__init__(self)
        self.type = None
        self.inputs = []
        self.outputs = []
        self.weights = []  # there could be overlap between inputs and weights
