def getPerm(ilayout: str, olayout: str):
    char2index = {}
    for i in range(len(ilayout)):
        c = ilayout[i]
        char2index[c] = i

    perm = [char2index[c] for c in olayout]
    return perm


def transform(input, ilayout: str, olayout: str):
    if (ilayout == olayout):
        return input

    perm = getPerm(ilayout, olayout)
    transfrom_axis = [input[p] for p in perm]
    return transfrom_axis


class Layout(object):
    def __init__(self, source: str, target: str):
        self.source = source
        self.target = target
        self.current = source

    def transform(self, input):
        output = transform(input, self.source, self.target)
        self.current = self.target
        return output

    @property
    def perm(self):
        return getPerm(self.source, self.target)

    def __str__(self):
        return self.current + '(' + self.source + '->' + self.target + ')'
