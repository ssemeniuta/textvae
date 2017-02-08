import numpy
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from nn.layers import Dropout
from nn.containers import Parallel, Sequential
from nn.models.base_model import BaseModel


class Sampler(object):

    def __init__(self, size):
        self.size = size
        self.srng = RandomStreams(seed=numpy.random.randint(1000000))

    def __call__(self, x):
        self.mu = x[:, :self.size]
        self.log_sigma = x[:, self.size:]

        eps = self.srng.normal(self.mu.shape)
        z = self.mu + T.exp(0.5 * self.log_sigma) * eps
        return z


class Dropword(Dropout):

    def __init__(self, p, dummy_word=0):
        super(Dropword, self).__init__(p)
        self.dummy = dummy_word

    def __call__(self, x):
        if self.train:
            mask = self.srng.binomial(x.shape, p=1 - self.p, dtype='int32')
            return x * mask + self.dummy * (1 - mask)
        return x


class Store(object):

    def __init__(self):
        self.stored = None

    def __call__(self, x):
        self.stored = x
        return x


class LMReconstructionModel(BaseModel):

    def __init__(self, layers, aux_loss=False, alpha=0.0, anneal=True):
        super(LMReconstructionModel, self).__init__(layers)
        self.input = T.imatrix()
        self.target = T.imatrix()
        self.step = theano.shared(0)
        self.anneal = anneal
        self.train = True
        self.aux_loss = aux_loss
        self.alpha = alpha
        self.anneal_start = 1000.0 if self.aux_loss else 10000.0
        self.anneal_end = self.anneal_start + 7000.0

    @property
    def costs(self):
        p = self.output(self.input)
        t = self.input.flatten()
        reconstruction_loss = T.nnet.categorical_crossentropy(p, t).reshape(self.input.shape).sum(axis=0)

        aux_reconstruction_loss = 0
        if self.aux_loss:
            l = self.layers[1]
            assert(isinstance(l, Parallel))
            l = l.branches[0].layers[-2]
            assert(isinstance(l, Parallel))
            l = l.branches[0].layers[-1]
            assert(isinstance(l, Store))
            p = l.stored
            aux_reconstruction_loss = T.nnet.categorical_crossentropy(p, t).reshape(self.input.shape).sum(axis=0)

        s = self.get_sampler()
        mu = s.mu
        log_sigma = s.log_sigma

        kld = 0.5 * T.sum(1 + log_sigma - mu ** 2 - T.exp(log_sigma), axis=1)
        eps = 0.001 if self.aux_loss else 0.0
        if self.anneal:
            kld_weight = T.clip((self.step - self.anneal_start) / (self.anneal_end - self.anneal_start), 0, 1 - eps) + eps
        else:
            kld_weight = 1

        if self.aux_loss:
            cost = T.mean(reconstruction_loss - kld * kld_weight + self.alpha * aux_reconstruction_loss)
        else:
            cost = T.mean(reconstruction_loss - kld * kld_weight)

        if self.aux_loss:
            return [cost, T.mean(reconstruction_loss), T.mean(kld), T.mean(aux_reconstruction_loss)]
        else:
            return [cost, T.mean(reconstruction_loss), T.mean(kld)]

    def get_sampler(self):
        for l in self.layers:
            if isinstance(l, Parallel):
                l = l.branches[0][-1]
            if isinstance(l, Sampler):
                return l
        raise Exception("sampler not found")

    def get_l2(self):
        return T.sum([T.sum(p**2) for p in self.params])

    def set_phase(self, train):
        super(LMReconstructionModel, self).set_phase(train)
        self.train = train

    @property
    def updates(self):
        upd = super(LMReconstructionModel, self).updates
        if self.train:
            upd[self.step] = self.step + 1
        return upd
