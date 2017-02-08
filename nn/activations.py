import theano.tensor as T


class Identity(object):

    def __call__(self, x):
        return x


class Tanh(object):

    def __call__(self, x):
        return T.tanh(x)


class Sigmoid(object):

    def __call__(self, x):
        return T.nnet.sigmoid(x)


class ReLU(object):

    def __call__(self, x):
        return T.maximum(x, 0)


class Maxout(object):

    def __init__(self, size, axis):
        self.size = size
        self.axis = axis

    def __call__(self, x):
        input_shape = tuple(x.shape)
        num_feature_maps = input_shape[self.axis]
        num_feature_maps_out = num_feature_maps // self.size

        pool_shape = (input_shape[:self.axis] +
                      (num_feature_maps_out, self.size) +
                      input_shape[self.axis+1:])

        x_reshaped = x.reshape(pool_shape)
        return T.max(x_reshaped, axis=self.axis + 1)


class Gated(object):

    def __init__(self, axis=1, inner_activation=T.tanh):
        self.axis = axis
        assert axis in [1, 2], "gated act does not support axis != 1, 2"
        self.inner_activation = inner_activation

    def __call__(self, x):
        if self.axis == 1:
            size = x.shape[1] / 2
            return self.inner_activation(x[:, :size]) * T.nnet.sigmoid(x[:, size:])
        if self.axis == 2:
            size = x.shape[2] / 2
            return self.inner_activation(x[:, :, :size]) * T.nnet.sigmoid(x[:, :, size:])
        raise Exception("err")
