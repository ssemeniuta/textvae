import numpy
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from collections import OrderedDict

from nn.activations import Sigmoid, Tanh, ReLU
from nn.initializers import Uniform
from nn.layers import Linear, Dropout
from nn.containers import Sequential
from nn.normalization import LayerNormalization


class LNLSTM(object):
    def __init__(self, input_size, layer_size, batch_size=1, p=0.0,
                 name="", activation=T.tanh, inner_activation=T.nnet.sigmoid, weight_init=Uniform(), persistent=False):

        self.h = theano.shared(numpy.zeros((batch_size, layer_size), dtype=theano.config.floatX), name=name+"_h_init")
        self.c = theano.shared(numpy.zeros((batch_size, layer_size), dtype=theano.config.floatX), name=name+"_c_init")

        self.params = []
        self.preact = Sequential([
            Linear(input_size+layer_size, layer_size * 4, weight_init=weight_init, name=name+"_ifog"),
            LayerNormalization(layer_size * 4, name=name + "_ln")
        ])
        self.params = self.preact.params

        self.dropout = Dropout(p)

        self.updates = []
        self.activation = activation
        self.inner_activation = inner_activation
        self.batch_size = batch_size
        self.layer_size = layer_size
        self.persistent = persistent

    def __call__(self, x):
        if self.persistent:
            outputs_info = [self.c, self.h]
        else:
            outputs_info = [T.zeros((x.shape[1], self.layer_size)), T.zeros((x.shape[1], self.layer_size))]

        [c, h], upd = theano.scan(self.step, x, outputs_info=outputs_info)
        if self.persistent:
            upd[self.c] = c[-1]
            upd[self.h] = h[-1]

        self.updates = OrderedDict()
        self.updates.update(upd)

        return h

    def step(self, x_t, c_tm1, h_tm1):
        ifog = self.preact(T.concatenate([x_t, h_tm1], axis=1))
        i_t, f_t, o_t, g_t = self._split(ifog)
        c_t = f_t * c_tm1 + i_t * self.dropout(g_t)
        h_t = o_t * self.activation(c_t)
        return c_t, h_t

    def set_phase(self, train):
        self.dropout.set_phase(train)

    def reset(self):
        if self.persistent:
            self.h.set_value(numpy.zeros_like(self.h.get_value(), dtype=theano.config.floatX))
            self.c.set_value(numpy.zeros_like(self.c.get_value(), dtype=theano.config.floatX))

    def _split(self, x):
        i = x[:, 0 * self.layer_size:1 * self.layer_size]
        f = x[:, 1 * self.layer_size:2 * self.layer_size]
        o = x[:, 2 * self.layer_size:3 * self.layer_size]
        g = x[:, 3 * self.layer_size:4 * self.layer_size]
        return self.inner_activation(i), self.inner_activation(f), self.inner_activation(o), self.activation(g)


class LNGRU(object):

    def __init__(self, input_size, layer_size, batch_size=1, name="", p=0.0, weight_init=Uniform(),
                 inner_activation=Sigmoid(), activation=Tanh(), persistent=False):
        self.activation = activation
        self.inner_activation = inner_activation
        self.layer_size = layer_size
        self.batch_size = batch_size
        self.persistent = persistent
        self.h = theano.shared(numpy.zeros((batch_size, layer_size), dtype=theano.config.floatX), name=name + "_h_init")

        self.rz = Sequential([
            Linear(input_size+layer_size, layer_size * 2, weight_init=weight_init, name=name+"_r"),
            LayerNormalization(layer_size * 2, name=name+"_ln_r"),
            inner_activation
        ])
        self.g = Sequential([
            Linear(input_size+layer_size, layer_size, weight_init=weight_init, name=name+"_g"),
            LayerNormalization(layer_size, name=name+"_ln_g"),
            activation,
            Dropout(p)
        ])

        self.params = self.rz.params + self.g.params

    def step(self, x, h_tm1):
        rz_t = self.rz(T.concatenate([x, h_tm1], axis=1))
        r_t = rz_t[:, :self.layer_size]
        z_t = rz_t[:, self.layer_size:]
        g_t = self.g(T.concatenate([x, r_t * h_tm1], axis=1))
        h_t = (1 - z_t) * h_tm1 + z_t * g_t
        return h_t

    def __call__(self, x):
        h_init = T.zeros((x.shape[1], self.layer_size))

        h, upd = theano.scan(self.step, sequences=x, outputs_info=[h_init])

        self.updates = upd
        if self.persistent:
            self.updates[self.h] = h[-1]

        return h

    def set_phase(self, train):
        self.g.set_phase(train)

    def reset(self):
        if self.persistent:
            self.h.set_value(numpy.zeros((self.batch_size, self.layer_size), dtype=theano.config.floatX))


class LNRNN(object):
    def __init__(self, input_size, layer_size, batch_size, p=0.0,
                 name="", activation=T.tanh, weight_init=Uniform(), persistent=False):
        self.h = theano.shared(numpy.zeros((batch_size, layer_size), dtype=theano.config.floatX), name=name+"_h_init")

        self.preact = Sequential([
            Linear(input_size+layer_size, layer_size, weight_init=weight_init, name=name+"_fc"),
            LayerNormalization(layer_size, name=name+"_ln"),
            activation,
            Dropout(p)
        ])
        self.params = self.preact.params

        self.activation = activation
        self.batch_size = batch_size
        self.layer_size = layer_size
        self.input_size = input_size
        self.persistent = persistent

    def __call__(self, x):
        h_init = self.h if self.persistent else T.zeros((x.shape[1], self.layer_size))
        h, upd = theano.scan(self.step, x, outputs_info=[h_init])
        if self.persistent:
            upd[self.h] = h[-1]

        self.updates = upd

        return h

    def step(self, x_t, h_tm1):
        h_t = self.preact(T.concatenate([x_t, h_tm1], axis=1))
        return h_t

    def set_phase(self, train):
        self.preact.set_phase(train)

    def reset(self):
        if self.persistent:
            self.h.set_value(numpy.zeros((self.batch_size, self.layer_size), dtype=theano.config.floatX))
