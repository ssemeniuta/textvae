import pickle
import theano
import theano.tensor as T
import numpy
from .. import utils
from .. import containers


class BaseModel(object):
    def __init__(self, layers):
        self.layers = containers.Sequential(layers)
        self.assert_unique_names()

    def output(self, x):
        return self.layers(x)

    def assert_unique_names(self):
        names = []
        for p in self.params:
            assert p.name not in names, "%s is not a unique name" % p.name
            names.append(p.name)

    def build(self, train):
        self.set_phase(train)
        costs = self.costs
        updates = self.updates
        return costs, updates

    @property
    def params(self):
        return self.layers.params

    @property
    def extra_params(self):
        return self.layers.extra_params

    @property
    def all_params(self):
        return self.layers.all_params

    @staticmethod
    def count_params(params):
        total = 0
        for p in params:
            shape = p.get_value(borrow=True).shape
            total += numpy.prod(shape)
        return total

    @property
    def total_params(self):
        return self.count_params(self.params)

    @property
    def total_extra_params(self):
        return self.count_params(self.extra_params)

    @property
    def total_all_params(self):
        return self.count_params(self.all_params)

    def reset(self):
        self.layers.reset()

    def set_phase(self, train):
        self.layers.set_phase(train)

    @property
    def updates(self):
        return self.layers.updates

    @property
    def costs(self):
        raise Exception("To be implemented in project specific models")

    def dump(self, filename):
        params = {}
        for p in self.all_params:
            if p.name in params:
                raise Exception("%s is not a unique name" % p.name)
            params[p.name] = p.get_value()

        flt = utils.FLT(filename, write=True)
        flt.dump(params)

    def load(self, filename, strict=True, silent=False):
        if not silent:
            print "loading %s" % filename
        flt = utils.FLT(filename)
        params = flt.load()
        self.set_params(params, strict, silent)

    def set_params(self, params, strict=True, silent=False):
        for p in self.all_params:
            if p.name not in params:
                if strict is False:
                    if silent is False:
                        print "not found %s" % p.name
                continue
            param = params[p.name]
            if param.shape != p.get_value().shape:
                msg = p.name + ", model shape = " + str(p.get_value().shape) + " loading shape = " + str(param.shape)
                if strict:
                    raise Exception(msg)
                else:
                    if silent is False:
                        print msg, "- skipping"
            else:
                p.set_value(param.astype(theano.config.floatX))
                if silent is False:
                    print "loaded %s" % p.name
