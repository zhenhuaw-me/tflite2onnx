from .common import logger

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



class Layout:
    def __init__(self, source: str, target: str):
        self.source = source
        self.target = target
        self.current = source

    def transform(self, input):
        logger.debug("Transforming from %s to %s...", selfsource, self.target)
        assert(self.current is not self.target)
        output = transform(input, self.source, self.target)
        self.current = target
        return output
