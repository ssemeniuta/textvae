import numpy
import theano
import theano.tensor as T
import theano.sandbox.cuda.dnn as dnn

from nn.containers import Sequential
from nn.initializers import Uniform
from nn.activations import Gated
from nn.normalization import BatchNormalization
from nn.layers import Dropout


class LayoutRNNToCNN(object):

    def __call__(self, x):
        assert x.ndim == 3, "layoutrnntocnn: expected 3 dims, got %d" % x.ndim
        x = x.dimshuffle((1, 2, 0))
        x = x[:, :, :, None]
        return x


class LayoutCNNToRNN(object):

    def __call__(self, x):
        assert x.ndim == 4, "layoutcnntornn: expected 4 dims, got %d" % x.ndim
        x = x[:, :, :, 0]
        x = x.dimshuffle((2, 0, 1))
        return x


class Pooling1d(object):

    def __init__(self, size, stride, pad=0, mode="max", glob=False):
        self.size = size
        self.stride = stride
        self.mode = mode
        self.pad = pad
        self.glob = glob

    def __call__(self, x):
        if self.glob:
            return dnn.dnn_pool(x, (x.shape[2], x.shape[3]), stride=(1, 1), mode=self.mode, pad=(0, 0))
        return dnn.dnn_pool(x, (self.size, 1), stride=(self.stride, 1), mode=self.mode, pad=(self.pad, 0))


class Convolution1d(object):

    def __init__(self, kernel_size, kernel_number, input_size, pad=0, causal=True, dilation=1,
                 weight_init=Uniform(), name="", keepdims=False, stride=1):
        w_shape = kernel_number, input_size, kernel_size, 1
        b_shape = kernel_number

        self.causal = causal
        self.dilation = dilation
        self.stride = stride
        if kernel_size == 1:
            self.causal = False
        mask = []
        for i in xrange(kernel_size / 2):
            mask.append(0)
        for i in xrange(kernel_size / 2 + 1):
            mask.append(1)

        mask = numpy.asarray(mask, dtype=theano.config.floatX)
        self.mask = theano.shared(mask)

        self.w = theano.shared(weight_init(w_shape), name="%s_W" % name)
        self.b = theano.shared(weight_init(b_shape), name="%s_b" % name)

        self.params = [self.w, self.b]
        self.keepdims = keepdims
        self.pad = pad

    def __call__(self, x):
        if x.ndim == 3:
            x = x[:, :, :, None]

        w = self.w
        if self.causal:
            mask = self.mask.dimshuffle('x', 'x', 0, 'x')
            w = w * mask
        out = T.nnet.conv2d(x, w, border_mode=(self.pad, 0), filter_dilation=(self.dilation, 1), subsample=(self.stride, 1))

        out += self.b.dimshuffle('x', 0, 'x', 'x')

        return out


class ResidualConvolution1d(object):

    def __init__(self, kernel_size, kernel_number, input_size, causal=True, dilation=1,
                 weight_init=Uniform(), name="", keepdims=False):

        assert kernel_number % 2 == 0
        assert kernel_size == 3

        self.conv = Sequential([
            Convolution1d(kernel_size, kernel_number, input_size,
                            pad=dilation, causal=causal, dilation=dilation,
                            weight_init=weight_init, name=name, keepdims=keepdims),
            BatchNormalization(kernel_number, collect=False, name=name+"_bn"),
            Gated(),
            Convolution1d(1, input_size, kernel_number / 2,
                            pad=0, causal=causal, keepdims=keepdims,
                            weight_init=weight_init, name=name + "_1x1"),
        ])
        self.params = self.conv.params

    def __call__(self, x):
        return x + self.conv(x)


class HighwayConvolution1d(object):

    def __init__(self, kernel_size, input_size, causal=True, dilation=1,
                 weight_init=Uniform(), name="", keepdims=False, p=0.0):
        from nn.normalization import LayerNormalization

        assert kernel_size == 3

        self.conv = Sequential([
            Convolution1d(kernel_size, input_size * 3, input_size,
                            pad=dilation, causal=causal, dilation=dilation,
                            weight_init=weight_init, name=name, keepdims=keepdims),
            BatchNormalization(input_size * 3, name=name+"_bn"),
        ])
        self.dropout = Dropout(p)
        self.input_size = input_size
        self.params = self.conv.params

    def __call__(self, x):
        i, f, g = self._split(self.conv(x))
        y = T.nnet.sigmoid(f) * x + T.nnet.sigmoid(i) * self.dropout(T.tanh(g))
        return y

    def _split(self, x):
        return x[:, 0 * self.input_size:1 * self.input_size], \
               x[:, 1 * self.input_size:2 * self.input_size], \
               x[:, 2 * self.input_size:3 * self.input_size]

    def set_phase(self, train):
        self.dropout.set_phase(train)


class Deconvolution1D(object):

    def __init__(self, num_channels, num_filters, filter_size, stride=1, causal=False,
                 pad=0, weight_init=Uniform(), name="", output_sz=-1):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.stride = stride
        self.pad = pad
        self.causal = causal
        self.output_sz = output_sz
        mask = []
        for i in xrange(filter_size / 2 + 1):
            mask.append(1)
        for i in xrange(filter_size / 2):
            mask.append(0)
        mask = numpy.asarray(mask, dtype=theano.config.floatX)
        self.mask = theano.shared(mask)

        w_shape = (num_channels, self.num_filters, self.filter_size, 1)
        self.W = theano.shared(weight_init(w_shape), name=name+"_W")
        self.b = theano.shared(weight_init(num_filters), name=name+"_b")

        self.params = [self.W, self.b]

    def __call__(self, x):
        from nn.utils import deconv_length
        if self.output_sz == -1:
            sz = deconv_length(x.shape[2], self.filter_size, self.stride, self.pad)
        else:
            sz = self.output_sz
        image = T.alloc(0., x.shape[0], self.num_filters, sz, 1)
        if self.causal:
            w = self.W * self.mask.dimshuffle('x', 'x', 0, 'x')
        else:
            w = self.W
        conved = dnn.dnn_conv(image, w, subsample=(self.stride, 1), border_mode=(self.pad, 0))

        grad = T.grad(conved.sum(), wrt=image, known_grads={conved: x})
        return grad + self.b.dimshuffle('x', 0, 'x', 'x')


