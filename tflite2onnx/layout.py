import logging

logger = logging.getLogger('tflite2onnx')


def transform(input, ilayout: str, olayout: str):
    if (ilayout == olayout):
        return input

    perm = getPerm(ilayout, olayout)
    transfrom_axis = [input[p] for p in perm]
    return transfrom_axis


def getPerm(ilayout: str, olayout: str):
    char2index = {}
    for i in range(len(ilayout)):
        c = ilayout[i]
        char2index[c] = i

    perm = [char2index[c] for c in olayout]
    return perm


class Layout(object):
    def __init__(self, source: str, target: str):
        self.source = source
        self.target = target
        self.current = source

    def markDone(self):
        assert(self.current != self.target)
        self.current = self.target

    def transform(self, input):
        logger.debug("Transforming from %s to %s...", self.source, self.target)
        # assert(self.current is not self.target)
        output = transform(input, self.source, self.target)
        self.current = self.target
        return output

    @property
    def match(self):
        return self.current == self.target

    def __str__(self):
        return self.current + '(' + self.source + '->' + self.target + ')'
