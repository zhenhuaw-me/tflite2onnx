from ..common import BaseABC


class Operator(BaseABC):
    type = None
    inputs = []
    outputs = []
