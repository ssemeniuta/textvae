import argparse
import numpy

from nn.layers1d import LayoutCNNToRNN, LayoutRNNToCNN, Convolution1d, HighwayConvolution1d
from nn.layers import Linear, Embed, Flatten, Reshape, SoftMax, Dropout, OneHot
from nn.activations import ReLU, Tanh, Gated
from nn.optimizer import Optimizer
from nn.updates import Adam
from nn.normalization import BatchNormalization
import nn.utils
from nn.models.lm_model import LMModel
from nn.databases.lm_database import LMDatabase


def make_model(n_classes, charcnn_size, charcnn_layers):
    layers = [
        OneHot(n_classes + 1),
        LayoutRNNToCNN(),
        Convolution1d(1, charcnn_size * 2, n_classes+1, name="decconvresize"),
        BatchNormalization(charcnn_size * 2, name="decbnresize"),
        Gated(),
    ]
    for i in xrange(charcnn_layers):
        layers.append(HighwayConvolution1d(3, charcnn_size, dilation=1, name="decconv%d" % i))
    layers.extend([
        LayoutCNNToRNN(),
        Linear(charcnn_size, n_classes, name="classifier"),
        SoftMax()
    ])

    model = LMModel(layers)

    return model


def main(lr, sample_size, charcnn_size, charcnn_layers):
    train_db = LMDatabase("train", batch_size=64, sample_size=sample_size)
    valid_db = LMDatabase("valid", sample_size=sample_size)

    n_classes = numpy.max(train_db.dataset) + 1

    model = make_model(n_classes, charcnn_size, charcnn_layers)

    out = nn.utils.forward(model, train_db)
    print out

    print model.total_params

    name = "charcnn.len_%d.charcnnsize_%d.charcnnlayers_%d" % (sample_size, charcnn_size, charcnn_layers)

    opt = Optimizer(model, train_db, valid_db, Adam(lr),
                    name=name, print_info=True)

    opt.train(100, decay_after=20, lr_decay=0.95)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-lr', default=0.001, type=float)
    parser.add_argument('-sample_size', default=56, type=int)
    parser.add_argument('-charcnn_size', default=256, type=int)
    parser.add_argument('-charcnn_layers', default=1, type=int)
    args = parser.parse_args()
    main(**vars(args))
