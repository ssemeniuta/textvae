import pickle
import theano
import theano.tensor as T
import numpy


class LMDatabase:
    def __init__(self, phase, batch_size=100, sample_size=10, augment=False):
        self.augment = augment
        self.batch_size = batch_size
        self.phase = phase
        self.dataset = numpy.load(open("data/lm.%s.npy" % phase))
        self.vocab_size = max(self.dataset) + 1
        print "%s: %d items" % (phase, len(self.dataset))
        self.sample_size = sample_size
        ins, outs, self.batch_number = self.create_dataset()
        self.ins = theano.shared(ins)
        self.outs = theano.shared(outs)
        self.index = T.iscalar()

    def set_data(self, data):
        print "%s: new data, %d items" % (self.phase, data.shape[0])
        self.dataset = data.copy()
        ins, outs, self.batch_number = self.create_dataset()
        self.ins.set_value(ins)
        self.outs.set_value(outs)

    def create_dataset(self):
        if self.augment:
            leftover = self.dataset.shape[0] % (self.sample_size * self.batch_size)
            if leftover == 0:
                leftover = self.sample_size * self.batch_size - 1
            start = numpy.random.randint(0, leftover-1)
            x_in = self.dataset[start:]
        else:
            x_in = self.dataset

        if x_in.shape[0] % (self.batch_size*self.sample_size) == 0:
            x_in = x_in[:-1]

        n_batches = x_in.shape[0] // (self.batch_size * self.sample_size)

        x = x_in[0:self.batch_size * self.sample_size * n_batches]
        x = x.reshape((self.batch_size, self.sample_size * n_batches))
        x = x.transpose((1, 0))

        t = x_in[1:self.batch_size * self.sample_size * n_batches + 1]
        t = t.reshape((self.batch_size, self.sample_size * n_batches))
        t = t.transpose((1, 0))

        return x.astype('int32'), t.astype('int32'), n_batches

    def total_batches(self):
        return self.batch_number

    def indices(self):
        for i in xrange(self.batch_number):
            yield i
        if self.augment:
            ins, outs, self.batch_number = self.create_dataset()
            self.ins.set_value(ins)
            self.outs.set_value(outs)

    def givens(self, x, t):
        return {
            x:  self.ins[self.sample_size * self.index: self.sample_size * (self.index + 1)],
            t: self.outs[self.sample_size * self.index: self.sample_size * (self.index + 1)]
        }
