import numpy
import theano
import theano.tensor as T
from nn.utils import Vocabulary


class TwitterReconstructionDatabase(object):

    def __init__(self, phase, batch_size, max_len=140, batches_per_epoch=1000, pad=True):
        self.phase = phase
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.max_len = max_len
        self.vocab = Vocabulary()
        self.vocab.add('<pad>')
        self.vocab.add('<unk>')
        self.vocab.add('<end>')
        for i in xrange(256):
            ch = chr(i)
            self.vocab.add(ch)
        self.n_classes = len(self.vocab)
        self.pad = pad

        self.tweets = []
        with open("data/tweets.txt") as f:
            while True:
                s = f.readline()
                if s == "":
                    break
                s = s.strip().split(" ")
                for i in xrange(len(s)):
                    if s[i].startswith('http://'):
                        s[i] = "url"
                    if s[i].startswith('https://'):
                        s[i] = "url"
                    if s[i].startswith("@"):
                        s[i] = "@userid"
                    #if s[i].startswith("#"):
                    #    s[i] = "#hashtag"

                s = ''.join([s1 + " " for s1 in s]).strip()
                tweet = s
                if len(tweet) <= max_len - 1:
                    self.tweets.append(tweet)
                if len(self.tweets) >= 1000000:
                    break

        valid_size = 10000
        if self.phase == 'train':
            self.tweets = self.tweets[valid_size:]
        else:
            self.tweets = self.tweets[:valid_size]

        print "%s: %d tweets, max %d chars" % (phase, len(self.tweets), max_len)

        x = self.make_batch()
        self.shared_x = theano.shared(x)

        self.index = T.iscalar()

    def to_inputs(self, tweet):
        chars = [self.vocab.by_word(ch, oov_word='<unk>') for ch in tweet]
        chars.append(self.vocab.by_word('<end>'))
        for i in xrange(self.max_len - len(tweet) - 1):
            chars.append(self.vocab.by_word('<pad>'))
        return numpy.asarray(chars)

    def make_batch(self):
        batch = numpy.zeros((self.max_len, self.batch_size))

        if self.pad:
            for i in xrange(self.batch_size):
                idx = numpy.random.randint(len(self.tweets))
                batch[:, i] = self.to_inputs(self.tweets[idx])
        else:
            idx = numpy.random.randint(len(self.tweets))
            max_len = len(self.tweets[idx])
            target_len = len(self.tweets[idx])
            batch[:, 0] = self.to_inputs(self.tweets[idx])
            i = 1
            while i < self.batch_size:
                idx = numpy.random.randint(len(self.tweets))
                if abs(len(self.tweets[idx]) - target_len) > 3:
                    continue
                batch[:, i] = self.to_inputs(self.tweets[idx])
                max_len = max(max_len, len(self.tweets[idx]) + 1)
                i += 1
            batch = batch[0:max_len]

        return batch.astype('int32')

    def total_batches(self):
        return self.batches_per_epoch

    def givens(self, x, t):
        return {
            x: self.shared_x[:, self.index * self.batch_size:(self.index+1) * self.batch_size],
        }

    def indices(self):
        for i in xrange(self.total_batches()):
            x = self.make_batch()
            self.shared_x.set_value(x)
            yield 0
