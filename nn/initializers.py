import numpy as np
import os.path
import theano


class Uniform(object):

    def __init__(self, scale=0.1):
        self.scale = scale

    def __call__(self, shape):
        return np.asarray(np.random.uniform(-self.scale, self.scale, shape), dtype=theano.config.floatX)


class Normal(object):

    def __init__(self, mean=0, scale=0.01):
        self.mean = mean
        self.scale = scale

    def __call__(self, shape):
        return np.asarray(np.random.normal(self.mean, self.scale, shape), dtype=theano.config.floatX)


class Zeros(object):

    def __call__(self, shape):
        return np.zeros(shape, dtype=theano.config.floatX)


class Constant(object):

    def __init__(self, value):
        self.value = value

    def __call__(self, shape):
        return np.ones(shape, dtype=theano.config.floatX) * self.value


class FromFile(object):

    def __init__(self, filename, backup=None):
        self.filename = filename
        self.backup = backup

    def __call__(self, shape):
        if os.path.exists(self.filename):
            return np.load(self.filename).astype(theano.config.floatX)
        if self.backup is not None:
            return self.backup(shape)
        raise Exception('%s not found, no backup provided' % self.filename)


class Orthogonal(object):

    def __init__(self, backup_init=Uniform()):
        self.backup_init = backup_init

    def __call__(self, shape):
        if isinstance(shape, tuple):
            if len(shape) == 2:
                """ benanne lasagne ortho init (faster than qr approach)"""
                flat_shape = (shape[0], np.prod(shape[1:]))
                a = np.random.normal(0.0, 1.0, flat_shape)
                u, _, v = np.linalg.svd(a, full_matrices=False)
                q = u if u.shape == flat_shape else v  # pick the one with the correct shape
                q = q.reshape(shape)
                return q[:shape[0], :shape[1]].astype(theano.config.floatX)
        return self.backup_init(shape)
