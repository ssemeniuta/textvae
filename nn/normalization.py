import theano
import theano.tensor as T

from nn.initializers import *


class BatchNormalization(object):

    def __init__(self, size, axis=1, epsilon=1e-4, alpha=0.1, name="", collect=True):
        self.epsilon = epsilon
        self.alpha = alpha
        self.train = True
        self.axis = axis
        self.collect = collect

        self.beta = theano.shared(Constant(0)(size), name=name+"_beta")
        self.gamma = theano.shared(Constant(1)(size), name=name+"_gamma")

        self.mean = theano.shared(Constant(0)(size), name=name+"_mean")
        self.inv_std = theano.shared(Constant(1)(size), name=name+"_inv_std")

        self.params = [self.beta, self.gamma]
        if self.collect:
            self.extra_params = [self.mean, self.inv_std]

    def __call__(self, x):
        axes = range(x.ndim)
        axes.remove(self.axis)
        axes = tuple(axes)
        input_mean = x.mean(axes)
        input_inv_std = T.inv(T.sqrt(x.var(axes) + self.epsilon))

        if self.train:
            mean = input_mean
            inv_std = input_inv_std
        else:
            if self.collect:
                mean = self.mean
                inv_std = self.inv_std
            else:
                mean = input_mean
                inv_std = input_inv_std

        self.updates = {}
        if self.train:
            if self.collect:
                self.updates[self.mean] = (1 - self.alpha) * self.mean + self.alpha * input_mean
                self.updates[self.inv_std] = (1 - self.alpha) * self.inv_std + self.alpha * input_inv_std

        # prepare dimshuffle pattern inserting broadcastable axes as needed
        param_axes = iter(range(x.ndim - len(axes)))
        pattern = ['x' if input_axis in axes
                   else next(param_axes)
                   for input_axis in range(x.ndim)]

        # apply dimshuffle pattern to all parameters
        beta = self.beta.dimshuffle(pattern)
        gamma = self.gamma.dimshuffle(pattern)
        mean = mean.dimshuffle(pattern)
        inv_std = inv_std.dimshuffle(pattern)

        # normalize
        normalized = (x - mean) * (gamma * inv_std) + beta
        return normalized

    def set_phase(self, train):
        self.train = train


class LayerNormalization(object):

    def __init__(self, size, axis=1, epsilon=1e-4, name=""):
        from nn.initializers import Constant
        self.epsilon = epsilon
        self.axis = axis

        self.beta = theano.shared(Constant(0)(size), name=name+"_beta")
        self.gamma = theano.shared(Constant(1)(size), name=name+"_gamma")

        self.params = [self.beta, self.gamma]

    def __call__(self, x):
        mean = x.mean(1, keepdims=True)
        inv_std = T.inv(T.sqrt(x.var(1, keepdims=True) + self.epsilon))

        pattern = ['x', 0] + ['x' for _ in xrange(x.ndim - 2)]
        beta = self.beta.dimshuffle(tuple(pattern))
        gamma = self.gamma.dimshuffle(tuple(pattern))

        # normalize
        normalized = (x - mean) * gamma * inv_std + beta
        return normalized
