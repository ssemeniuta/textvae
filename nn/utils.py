import theano
import numpy
import json
import os.path


def forward(model, db, train=False, out=None, full=False, upd=None):
    if upd is None:
        upd = []
    if out is None:
        out, upd = model.build(train=train)
    f = theano.function([db.index], out,
                        givens=db.givens(model.input, model.target),
                        updates=upd,
                        on_unused_input='ignore')
    for idx in db.indices():
        out = f(idx)
        if full is False:
            break

    model.reset()
    return out


def save_json(filename, data):
    s = json.JSONEncoder().encode(data)
    with open(filename, 'w') as f:
        print >>f, s


def read_json(filename):
    with open(filename) as f:
        s = f.read()
    data = json.JSONDecoder().decode(s)
    return data


def read_json_as_tuple(filename, keys):
    data = read_json(filename)
    data = tuple([data[key] for key in keys])
    return data


class FLT(object):

    def __init__(self, filename=None, write=False, f=None):
        if f is not None:
            self.f = f
            return
        self.f = None
        self.open(filename, write)

    def save_array(self, data):
        float_data = data.astype(numpy.float32)
        print >> self.f, "#%d" % data.ndim
        for i in xrange(data.ndim):
            print >> self.f, "#%d" % data.shape[i]
        print >> self.f, "#float"
        self.f.write(float_data.tostring())
        print >> self.f

    def dump(self, params, close=True):
        for p in params:
            print >> self.f, "#%s" % p
            self.save_array(params[p])
        if close:
            self.close()

    def load(self):
        params = {}
        while True:
            c = self.f.read(1)
            if c == '':
                break
            assert c == "#", 'incorrect param name format'
            name = self.f.readline().strip()
            arr = self.load_array()
            if name not in params:
                params[name] = arr
            else:
                raise Exception('%d param already loaded' % name)
        return params

    def load_array(self):
        c = self.f.read(1)
        assert c == "#", 'incorrect param ndim format'
        ndim = self.f.readline().strip()
        ndim = int(ndim)
        dims = []
        for i in xrange(ndim):
            c = self.f.read(1)
            assert c == "#", 'incorrect param dim format'
            dim = self.f.readline().strip()
            dim = int(dim)
            dims.append(dim)
        dtype = self.f.readline().strip()
        assert dtype == "#float", 'incorrect dtype'
        bytes = numpy.prod(dims) * 4
        if ndim == 0:
            bytes = 4
        bytes = self.f.read(bytes)
        arr = numpy.fromstring(bytes, dtype=numpy.float32)
        c = self.f.read(1)
        return arr.reshape(dims)

    def close(self):
        self.f.close()

    def open(self, filename, write=False):
        mode = 'w' if write else 'r'
        self.f = open(filename, mode)


class Vocabulary(object):

    def __init__(self):
        self.word_to_index = {}
        self.index_to_word = {}
        self.idx = 0

    def add(self, word):
        if word in self.word_to_index:
            return self.word_to_index[word]
        self.word_to_index[word] = self.idx
        self.index_to_word[self.idx] = word
        self.idx += 1
        return self.idx - 1

    def by_word(self, word, oov_word=None):
        if word in self.word_to_index:
            return self.word_to_index[word]
        if oov_word is not None:
            assert oov_word in self.word_to_index
            return self.word_to_index[oov_word]
        return -1

    def by_index(self, index):
        return self.index_to_word[index]

    def by_idx(self, idx):
        return self.by_index(idx)

    def __len__(self):
        assert len(self.index_to_word) == len(self.word_to_index)
        return len(self.index_to_word)


def update_vars_as_dict(param_updates):
    try:
        _ = param_updates.vars
    except AttributeError:
        print "update_vars_as_dict: can not find update variables"
        return {}

    v = {}
    for var in param_updates.vars:
        if var.name in v:
            raise Exception("%s is not a unique name" % p.name)
        v[var.name] = var.get_value()

    return v


def dump_update_vars(param_updates, filename):
    v = update_vars_as_dict(param_updates)
    flt = FLT(filename, write=True)
    flt.dump(v)


def load_update_vars(param_updates, filename):
    try:
        _ = param_updates.vars
    except AttributeError:
        print "load_update_vars: can not find update variables"
        return

    flt = FLT(filename)
    v = flt.load()
    for var in param_updates.vars:
        if var.name not in v:
            raise Exception("can not find variable %s" % var.name)
        var.set_value(v[var.name].astype(theano.config.floatX))


def load_bin_vec(fname, words):
  print "loading", fname
  vocab = set(words)
  word_vecs = {}
  with open(fname, "rb") as f:
    header = f.readline()
    vocab_size, layer1_size = map(int, header.split())
    binary_len = numpy.dtype('float32').itemsize * layer1_size
    print 'vocab_size, layer1_size', vocab_size, layer1_size
    count = 0
    for i, line in enumerate(xrange(vocab_size)):
      if i % 100000 == 0:
        print '.',
      word = []
      while True:
        ch = f.read(1)
        if ch == ' ':
            word = ''.join(word)
            break
        if ch != '\n':
            word.append(ch)
      if word in vocab:
        count += 1
        word_vecs[word] = numpy.fromstring(f.read(binary_len), dtype='float32')
      else:
          f.read(binary_len)
    print "done"
    print "words found in wor2vec embeddings: ", count
    return word_vecs


def load_txt_vec(fname, words):
    print "loading", fname
    word_vecs = {}
    vocab = set(words)
    lines_done = 0
    words_found = 0

    with open(fname) as f:
        while True:
            line = f.readline()
            if line == "":
                break
            line = line.strip().split(' ')
            if line[0] in vocab:
                words_found += 1
                vec = [float(val) for val in line[1:]]
                vec = numpy.asarray(vec)
                word_vecs[line[0]] = vec
            lines_done += 1
            if lines_done % 100 == 0:
                print "\r%d %d" % (lines_done, words_found),

    print "done, found %d words" % words_found
    return word_vecs


def make_embeddings(vocab, emb_size, weight_init=None, emb_filename=None):
    from initializers import Uniform
    if weight_init is None:
        weight_init = Uniform()
    emb = weight_init((len(vocab), emb_size))

    if emb_filename is None:
        return emb

    assert isinstance(vocab, Vocabulary)
    words = vocab.word_to_index.keys()
    _, ext = os.path.splitext(emb_filename)
    if ext == ".bin":
        word_vecs = load_bin_vec(emb_filename, words)
    elif ext == ".txt":
        word_vecs = load_txt_vec(emb_filename, words)
    else:
        raise Exception("Unknown embedding format: %s" % emb_filename)

    for w in words:
        idx = vocab.by_word(w)
        if w in word_vecs:
            emb[idx] = word_vecs[w]

    return emb


def deconv_length(output_length, filter_size, stride, pad=0):
    if output_length is None:
        return None

    output_length = output_length * stride
    if pad == 'valid':
        input_length = output_length + filter_size - 1
    elif pad == 'full':
        input_length = output_length - filter_size + 1
    elif pad == 'same':
        input_length = output_length
    elif isinstance(pad, int):
        input_length = output_length - 2 * pad + filter_size - 1
    else:
        raise ValueError('Invalid pad: {0}'.format(pad))

    return input_length


class Callback(object):

    def __init__(self, idx=0, comparator=None):
        self.best_costs = None
        self.comparator = comparator
        if comparator is None:
            self.comparator = lambda x, y: x[idx] < y[idx]

    def __call__(self, costs):
        best = True
        if self.best_costs is not None:
            best = self.comparator(costs, self.best_costs)
        if best:
            self.best_costs = costs
        return best

