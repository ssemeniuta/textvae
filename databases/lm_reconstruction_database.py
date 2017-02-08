import numpy
import theano
import theano.tensor as T
import pickle


class LmReconstructionDatabase(object):

    def __init__(self, phase, batch_size=64, batches_per_epoch=10000, sample_size=200, random_samples=True):
        self.phase = phase
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.sample_size = sample_size
        self.data = numpy.load('data/lm.%s.npy' % phase)
        self.vocab = pickle.load(open('data/char_vocab.pkl'))
        self.n_classes = numpy.max(self.data) + 2
        self.pad_word = self.n_classes - 1
        self.random_samples = random_samples

        x = self.make_batch()
        self.shared_x = theano.shared(x)

        self.index = T.iscalar()

    def total_batches(self):
        return self.batches_per_epoch

    def givens(self, x, t):
        return {
            x: self.shared_x[:, self.index * self.batch_size:(self.index+1) * self.batch_size],
        }

    def get_sample(self):
        newline = self.vocab.by_word("\n")
        space = self.vocab.by_word(" ")
        c = int(self.sample_size / 4 * 3)
        while True:
            idx = numpy.random.randint(0, self.data.shape[0] - 2 * self.sample_size)
            s = self.data[idx:idx+c]
            newline_idx = numpy.where(s == newline)
            if len(newline_idx[0]) == 0:
                break

        if self.data[idx] != space:
            s = self.data[idx:idx+self.sample_size]
            space_idx = numpy.where(s == space)
            idx = idx + space_idx[0][0]

        s = self.data[idx+1:idx + self.sample_size+1].copy()
        newline_idx = numpy.where(s == newline)
        if len(newline_idx[0]) != 0:
            s[newline_idx[0][0]:] = self.pad_word
            end = newline_idx[0][0] - 1
        else:
            end = self.sample_size - 1

        if s[end] != space:
            space_idx = numpy.where(s == space)[0][-1]
            s[space_idx+1:] = self.pad_word

        return s

    def make_batch(self):
        indices = numpy.random.randint(0, self.data.shape[0] - self.sample_size, self.batch_size)
        x = numpy.zeros((self.sample_size, self.batch_size))
        for i in xrange(self.batch_size):
            if self.random_samples:
                idx = indices[i]
                x[:, i] = self.data[idx:idx+self.sample_size]
            else:
                sample = self.get_sample()
                x[:, i] = sample

        return x.astype('int32')

    def indices(self):
        for i in xrange(self.batches_per_epoch):
            x = self.make_batch()
            self.shared_x.set_value(x)
            yield 0
