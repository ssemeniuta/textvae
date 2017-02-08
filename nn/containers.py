from collections import OrderedDict


class Sequential(object):

    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def __getitem__(self, idx):
        return self.layers[idx]

    def __setitem__(self, idx, value):
        self.layers[idx] = value

    def get_params(self, extra=False):
        p = []
        for l in self.layers:
            try:
                if extra:
                    p.extend(l.extra_params)
                else:
                    p.extend(l.params)
            except AttributeError:
                pass
        seen = set()
        p = [x for x in p if not (x in seen or seen.add(x))]
        return p

    @property
    def params(self):
        return self.get_params(extra=False)

    @property
    def extra_params(self):
        return self.get_params(extra=True)

    @property
    def all_params(self):
        return self.params + self.extra_params

    def set_phase(self, train):
        for l in self.layers:
            try:
                l.set_phase(train)
            except AttributeError:
                pass

    @property
    def updates(self):
        res = OrderedDict()
        for l in self.layers:
            try:
                upd = l.updates
                for var in upd:
                    if var in res:
                        print "update for %s already present, overriding" % var.name
                    res[var] = upd[var]
            except AttributeError:
                pass
        return res

    def reset(self):
        for l in self.layers:
            try:
                l.reset()
            except AttributeError:
                pass


class Parallel(object):

    def __init__(self, branches, shared_input=True, concat_axis=-1):
        self.axis = concat_axis
        self.shared_input = shared_input
        self.branches = []
        self.params = []
        self.extra_params = []
        for branch in branches:
            c = Sequential(branch)
            self.branches.append(c)
            self.params.extend(c.params)
            self.extra_params.extend(c.extra_params)

    def reset(self):
        for b in self.branches:
            b.reset()

    def set_phase(self, train):
        for b in self.branches:
            b.set_phase(train)

    @property
    def updates(self):
        res = OrderedDict()
        for b in self.branches:
            try:
                upd = b.updates
                for var in upd:
                    if var in res:
                        print "update for %s already present, overriding" % var.name
                    res[var] = upd[var]
            except AttributeError:
                pass
        return res

    def __call__(self, x):
        import theano.tensor as T
        y = []
        for i in xrange(len(self.branches)):
            if self.shared_input:
                y.append(self.branches[i](x))
            else:
                y.append(self.branches[i](x[i]))

        if self.axis != -1:
            y = T.concatenate(y, axis=self.axis)

        return y
