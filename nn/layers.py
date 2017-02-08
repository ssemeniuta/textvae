import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import theano.sandbox.cuda.dnn as dnn
import numpy as np

from initializers import Uniform
from activations import *


class SoftMax(object):

    def __init__(self, keepdims=False):
        self.keepdims = keepdims

    def __call__(self, x):
        y = T.nnet.softmax(x.reshape((-1, x.shape[-1])))
        if self.keepdims:
            y = y.reshape(x.shape)
        return y


class Linear(object):

    def __init__(self, input_size, output_size, weight_init=Uniform(), name="", biases=True):
        self.W = theano.shared(weight_init((input_size, output_size)), name="%s_W" % name)
        self.b = None
        self.params = [self.W]

        if biases:
            self.b = theano.shared(weight_init(output_size), name="%s_b" % name)
            self.params.append(self.b)

    def __call__(self, x):
        y = T.dot(x, self.W)
        if self.b is not None:
            y += self.b
        return y


class Embed(object):
    def __init__(self, input_size, output_size, weight_init=Uniform(), learnable=True, name=""):
        self.W = theano.shared(weight_init((input_size, output_size)), name=name+"_W")
        if learnable:
            self.params = [self.W]
        else:
            self.params = []

    def __call__(self, x):
        return self.W[x]


class Dropout(object):
    def __init__(self, p):
        self.srng = RandomStreams(seed=np.random.randint(1000000))
        self.p = p
        self.train = True

    def __call__(self, x):
        if self.p == 0.0:
            return x
        if self.train:
            return x * self.srng.binomial(x.shape, p=1-self.p, dtype=theano.config.floatX) / (1 - self.p)
        return x

    def set_phase(self, train):
        self.train = train


class Dimshuffle(object):

    def __init__(self, new_axes):
        self.new_axes = new_axes

    def __call__(self, x):
        return x.dimshuffle(self.new_axes)


class Reshape(object):

    def __init__(self, new_shape):
        self.new_shape = new_shape

    def __call__(self, x):
        return x.reshape(self.new_shape)


class Flatten(object):

    def __call__(self, x):
        return x.reshape((x.shape[0], -1))


class OneHot(object):

    def __init__(self, n_classes):
        self.n_classes = n_classes

    def __call__(self, x):
        import theano.tensor.extra_ops as extra_ops
        y = extra_ops.to_one_hot(x.flatten(), self.n_classes)
        if x.ndim == 1:
            return y
        return y.reshape((x.shape[0], x.shape[1], -1))


