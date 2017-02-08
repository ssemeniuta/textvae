import pickle
import argparse
from scipy.stats import norm
import numpy
import theano
import theano.tensor as T
from databases.twitter_reconstruction_database import TwitterReconstructionDatabase

from nn.containers import Sequential
from nn.rnns import LNLSTM
from nn.layers import OneHot
from nn.utils import Vocabulary

from lm_vae import Sampler
from lm_vae_sample import LNLSTMStep
from twitter_vae_charlevel import make_model


def main(z, sample_size, p, lstm_size, mode, alpha):
    vocab = Vocabulary()
    vocab.add('<pad>')
    vocab.add('<unk>')
    vocab.add('<end>')
    for i in xrange(256):
        ch = chr(i)
        vocab.add(ch)
    n_classes = len(vocab)

    model = make_model(z, sample_size, p, n_classes, lstm_size, alpha)
    name = "twittervae.charlevel.z_%d.len_%d.p_%.2f.lstmsz_%d.alpha_%.2f" % (z, sample_size, p, lstm_size, alpha)
    model.load("exp/%s/model.flt" % name)
    model.set_phase(train=False)

    start_word = n_classes

    if mode == 'vary':
        n = 7
        sampled = numpy.random.normal(0, 1, (1, z))
        sampled = numpy.repeat(sampled, n * z, axis=0)
        for dim in xrange(z):
            eps = 0.01
            x = numpy.linspace(eps, 1 - eps, num=n)
            x = norm.ppf(x)
            sampled[dim*n:(dim+1)*n, dim] = x
        n *= z
    elif mode == 'interpolatereal':
        valid_db = TwitterReconstructionDatabase("valid", 50, batches_per_epoch=100, max_len=sample_size)
        s1 = numpy.random.randint(0, len(valid_db.tweets))
        s2 = numpy.random.randint(0, len(valid_db.tweets))
        encoder = model.layers[0].branches[0]
        sampler = encoder[-1]
        assert isinstance(sampler, Sampler)
        ins = numpy.zeros((sample_size, 2))
        ins[:, 0] = valid_db.to_inputs(valid_db.tweets[s1])
        ins[:, 1] = valid_db.to_inputs(valid_db.tweets[s2])
        x = T.imatrix()
        z = encoder(x)
        mu = sampler.mu
        f = theano.function([x], mu)
        z = f(ins.astype('int32'))
        s1_z = z[0]
        s2_z = z[1]
        n = 7
        s1_z = numpy.repeat(s1_z[None, :], n, axis=0)
        s2_z = numpy.repeat(s2_z[None, :], n, axis=0)
        steps = numpy.linspace(0, 1, n)[:, None]
        sampled = s1_z * (1 - steps) + s2_z * steps
    elif mode == 'arithm':
        valid_db = TwitterReconstructionDatabase("valid", 50, batches_per_epoch=100, max_len=sample_size)
        s1 = numpy.random.randint(0, len(valid_db.tweets))
        s2 = numpy.random.randint(0, len(valid_db.tweets))
        s3 = numpy.random.randint(0, len(valid_db.tweets))
        print valid_db.tweets[s1]
        print valid_db.tweets[s2]
        print valid_db.tweets[s3]
        encoder = model.layers[0].branches[0]
        sampler = encoder[-1]
        assert isinstance(sampler, Sampler)
        ins = numpy.zeros((sample_size, 3))
        ins[:, 0] = valid_db.to_inputs(valid_db.tweets[s1])
        ins[:, 1] = valid_db.to_inputs(valid_db.tweets[s2])
        ins[:, 2] = valid_db.to_inputs(valid_db.tweets[s3])
        x = T.imatrix()
        z = encoder(x)
        mu = sampler.mu
        f = theano.function([x], mu)
        z = f(ins.astype('int32'))
        s1_z = z[0]
        s2_z = z[1]
        s3_z = z[1]
        n = 1
        sampled = s1_z - s2_z + s3_z
        sampled = sampled[None, :]
    elif mode == 'interpolate':
        z = numpy.random.normal(0, 1, (2, z))
        s1_z = z[0]
        s2_z = z[1]
        n = 7
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
    from_z = decoder_from_z(sampled)

    layers = model.layers[-3:]
    layers[0] = LNLSTMStep(layers[0])
    step = Sequential(layers)
    embed = model.layers[1].branches[1].layers[-1]

    words = start_words
    generated = []
    for i in xrange(sample_size):
        ins = T.concatenate([from_z[i], embed(words)], axis=1)
        pred = step(ins)
        words = T.argmax(pred, axis=1)
        generated.append(words[None, :])

    generated = T.concatenate(generated, axis=0)
    import time
    t = time.time()
    print "compiling...",
    f = theano.function([], outputs=generated)
    print "done, took %f secs" % (time.time() - t)
    w = f()

    results = []

    pad = vocab.by_word("<pad>")
    end = vocab.by_word("<end>")
    for i in xrange(w.shape[1]):
        s = []
        for idx in w[:, i]:
            if idx == end:
                break
            if idx == pad:
                break
            s.append(vocab.by_index(idx))
        r = ''.join(s)
        if mode == "vary":
            if i % n == 0:
                print "dimension %d" % (i / n)
        print r.strip()
        results.append(r)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-z', default=100, type=int)
    parser.add_argument('-p', default=0.0, type=float)
    parser.add_argument('-alpha', default=0.2, type=float)
    parser.add_argument('-sample_size', default=128, type=int)
    parser.add_argument('-lstm_size', default=1000, type=int)
    parser.add_argument('-mode', default="sample")
    args = parser.parse_args()
    main(**vars(args))
