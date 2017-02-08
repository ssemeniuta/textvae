import pickle
import argparse
from scipy.stats import norm
import numpy
import theano
import theano.tensor as T
from databases.lm_reconstruction_database import LmReconstructionDatabase

from nn.containers import Sequential
from nn.rnns import LNLSTM
from nn.layers import OneHot
from lm_vae_lstm import make_model, Sampler


class LNLSTMStep(object):

    def __init__(self, lstm):
        assert isinstance(lstm, LNLSTM)
        self.lstm = lstm
        self.c = None
        self.h = None

    def reset_state(self):
        self.c = None
        self.h = None

    def __call__(self, x):
        if self.c is None:
            self.c = T.zeros((x.shape[0], self.lstm.layer_size))
            self.h = T.zeros((x.shape[0], self.lstm.layer_size))
        self.c, self.h = self.lstm.step(x, self.c, self.h)
        return self.h


def to_inputs(s, vocab, sample_size):
    assert len(s) <= sample_size
    s = [vocab.by_word(w) for w in s]
    for i in xrange(sample_size - len(s)):
        s.append(len(vocab.word_to_index))
    return numpy.asarray(s)


def main(z, sample_size, p, encdec_layers, lstm_size, pad_string, mode, alpha):
    vocab = pickle.load(open("data/char_vocab.pkl"))

    train_db = LmReconstructionDatabase("train", batches_per_epoch=1000, sample_size=sample_size, random_samples=False)
    valid_db = LmReconstructionDatabase("valid", batches_per_epoch=100, sample_size=sample_size, random_samples=False)

    model = make_model(z, sample_size, p, train_db.n_classes, encdec_layers, lstm_size, alpha)
    name = "lm.charvae.z_%d.len_%d.layers_%d.p_%.2f.alpha_%.2f.lstmsz_%d" % \
           (z, sample_size, encdec_layers, p, alpha, lstm_size)
    model.load("exp/%s/model.flt" % name)
    model.set_phase(train=False)

    start_word = train_db.n_classes

    if mode == 'manifold':
        assert z == 2
        steps = 10
        eps = 0.001
        x = numpy.linspace(eps, 1 - eps, num=steps)
        y = numpy.linspace(eps, 1 - eps, num=steps)
        n = steps ** 2
        xy = [(i, j) for i in x for j in y]
        xy = numpy.asarray(xy)
        sampled = norm.ppf(xy)
    elif mode == 'vary':
        dim = numpy.random.randint(z)
        print "dimension %d" % dim
        s = "<unk> caller to a local radio station said cocaine"
        s = to_inputs(s, vocab, sample_size)
        encoder = model.layers[0].branches[0]
        sampler = encoder[-1]
        assert isinstance(sampler, Sampler)
        ins = s[:, None]
        x = T.imatrix()
        z = encoder(x)
        mu = sampler.mu
        f = theano.function([x], mu)
        z = f(ins.astype('int32'))
        s_z = z[0]
        n = 15
        eps = 0.001
        x = numpy.linspace(eps, 1 - eps, num=n)
        x = norm.ppf(x)
        sampled = numpy.repeat(s_z[None, :], n, axis=0)
        sampled[:, dim] = x
    elif mode == 'interpolate':
        s1 = "<unk> caller to a local radio station said cocaine"
        s2 = "giving up some of its gains as the dollar recovered"
        s1 = to_inputs(s1, vocab, sample_size)
        s2 = to_inputs(s2, vocab, sample_size)
        encoder = model.layers[0].branches[0]
        sampler = encoder[-1]
        assert isinstance(sampler, Sampler)
        ins = numpy.zeros((sample_size, 2))
        ins[:, 0] = s1
        ins[:, 1] = s2
        x = T.imatrix()
        z = encoder(x)
        mu = sampler.mu
        f = theano.function([x], mu)
        z = f(ins.astype('int32'))
        s1_z = z[0]
        s2_z = z[1]
        n = 15
        s1_z = numpy.repeat(s1_z[None, :], n, axis=0)
        s2_z = numpy.repeat(s2_z[None, :], n, axis=0)
        steps = numpy.linspace(0, 1, n)[:, None]
        sampled = s1_z * (1 - steps) + s2_z * steps
    else:
        n = 100
        sampled = numpy.random.normal(0, 1, (n, z))

    start_words = numpy.ones(n) * start_word
    start_words = theano.shared(start_words.astype('int32'))
    sampled = theano.shared(sampled.astype(theano.config.floatX))

    decoder_from_z = model.layers[1].branches[0]
    from_z = decoder_from_z(sampled.astype(theano.config.floatX))

    layers = model.layers[-3:]
    layers[0] = LNLSTMStep(layers[0])
    step = Sequential(layers)
    onehot = OneHot(train_db.n_classes+2)

    words = start_words
    generated = []
    for i in xrange(sample_size):
        ins = T.concatenate([from_z[i], onehot(words)], axis=1)
        pred = step(ins)
        words = T.argmax(pred, axis=1)
        generated.append(words[None, :])

    generated = T.concatenate(generated, axis=0)
    f = theano.function([], outputs=generated)
    w = f()

    if pad_string not in vocab.word_to_index:
        vocab.add(pad_string)
    else:
        raise Exception("%s is already in the vocabulary" % pad_string)

    results = []

    for i in xrange(w.shape[1]):
        s = [vocab.by_index(idx) for idx in w[:, i]]
        r = ''.join(s)
        print r
        results.append(r)

    if mode == 'manifold':
        lines = 3
        steps = int(numpy.sqrt(n))
        for i in xrange(steps):
            for k in xrange(lines):
                for j in xrange(steps):
                    r = results[i * steps + j]
                    l = len(r) / lines
                    print r[k*l:(k+1)*l], '  ',
                print
            print

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-z', default=2, type=int)
    parser.add_argument('-lr', default=0.001, type=float)
    parser.add_argument('-p', default=0.2, type=float)
    parser.add_argument('-sample_size', default=52, type=int)
    parser.add_argument('-encdec_layers', default=2, type=int)
    parser.add_argument('-lstm_size', default=500, type=int)
    parser.add_argument('-pad_string', default="~")
    parser.add_argument('-mode', default="sample")
    args = parser.parse_args()
    main(**vars(args))
