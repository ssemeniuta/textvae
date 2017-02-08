import argparse
import numpy
import theano
import theano.tensor as T
from databases.lm_reconstruction_database import LmReconstructionDatabase

from vae import Sampler, Dropword, Store, LMReconstructionModel
from nn.layers1d import LayoutCNNToRNN, LayoutRNNToCNN, Convolution1d, Deconvolution1D, HighwayConvolution1d
from nn.layers import Linear, Embed, Flatten, Reshape, SoftMax, Dropout, OneHot
from nn.activations import ReLU, Tanh, Gated
from nn.optimizer import Optimizer
from nn.updates import Adam
from nn.rnns import LNLSTM
from nn.containers import Sequential, Parallel
from nn.normalization import BatchNormalization
import nn.utils


def make_model(z, sample_size, dropword_p, n_classes, encdec_layers, lstm_size, alpha):
    assert sample_size % (2 ** encdec_layers) == 0
    if encdec_layers == 2:
        encoder = [
            OneHot(n_classes),
            LayoutRNNToCNN(),
            Convolution1d(3, 128, n_classes, pad=1, stride=2, causal=False, name="conv1"),
            BatchNormalization(128, name="bn1", collect=False),
            ReLU(),
            Convolution1d(3, 256, 128, pad=1, stride=2, causal=False, name="conv2"),
            BatchNormalization(256, name="bn2", collect=False),
            ReLU(),
            Flatten(),
            Linear(sample_size / 4 * 256, z * 2, name="fc_encode"),
            Sampler(z),
        ]
        decoder_from_z = [
            Linear(z, sample_size / 4 * 256, name="fc_decode"),
            ReLU(),
            Reshape((-1, 256, sample_size / 4, 1)),
            Deconvolution1D(256, 128, 3, pad=1, stride=2, name="deconv2"),
            BatchNormalization(128, name="deconv_bn2", collect=False),
            ReLU(),
            Deconvolution1D(128, 200, 3, pad=1, stride=2, name="deconv1"),
            BatchNormalization(200, name="deconv_bn1", collect=False),
            ReLU(),
            LayoutCNNToRNN(),
            Parallel([
                [
                    Linear(200, n_classes, name="aux_classifier"),
                    SoftMax(),
                    Store()
                ],
                []
            ], shared_input=True),
            lambda x: x[1]
        ]
    elif encdec_layers == 3:
        encoder = [
            OneHot(n_classes),
            LayoutRNNToCNN(),
            Convolution1d(3, 128, n_classes, pad=1, stride=2, causal=False, name="conv1"),
            BatchNormalization(128, name="bn1"),
            ReLU(),
            Convolution1d(3, 256, 128, pad=1, stride=2, causal=False, name="conv2"),
            BatchNormalization(256, name="bn2"),
            ReLU(),
            Convolution1d(3, 512, 256, pad=1, stride=2, causal=False, name="conv3"),
            BatchNormalization(512, name="bn3"),
            ReLU(),
            Flatten(),
            Linear(sample_size / 8 * 512, z * 2, name="fc_encode"),
            Sampler(z),
        ]
        decoder_from_z = [
            Linear(z, sample_size / 8 * 512, name="fc_decode"),
            ReLU(),
            Reshape((-1, 512, sample_size / 8, 1)),
            Deconvolution1D(512, 256, 3, pad=1, stride=2, name="deconv3"),
            BatchNormalization(256, name="deconv_bn3", collect=False),
            ReLU(),
            Deconvolution1D(256, 128, 3, pad=1, stride=2, name="deconv2"),
            BatchNormalization(128, name="deconv_bn2", collect=False),
            ReLU(),
            Deconvolution1D(128, 200, 3, pad=1, stride=2, name="deconv1"),
            BatchNormalization(200, name="deconv_bn1", collect=False),
            ReLU(),
            LayoutCNNToRNN(),
            Parallel([
                [
                    Linear(200, n_classes, name="aux_classifier"),
                    SoftMax(),
                    Store()
                ],
                []
            ], shared_input=True),
            lambda x: x[1]
        ]
    else:
        raise Exception("unsupported number of encdec layers %d" % encdec_layers)

    start_word = n_classes
    dummy_word = n_classes + 1
    decoder_from_words = [
        Dropword(dropword_p, dummy_word=dummy_word),
        lambda x: T.concatenate([T.ones((1, x.shape[1]), dtype='int32') * start_word, x], axis=0),
        lambda x: x[:-1],
        OneHot(n_classes+2),
    ]
    layers = [
        Parallel([
            encoder,
            []
        ], shared_input=True),
        Parallel([
            decoder_from_z,
            decoder_from_words
        ], shared_input=False),
        lambda x: T.concatenate(x, axis=2),
        LNLSTM(200+n_classes+2, lstm_size, name="declstm"),
        Linear(lstm_size, n_classes, name="classifier"),
        SoftMax(),
    ]

    model = LMReconstructionModel(layers, aux_loss=True, alpha=alpha)

    return model


def main(z, lr, sample_size, p, encdec_layers, lstm_size, alpha, anneal):
    train_db = LmReconstructionDatabase("train", batches_per_epoch=1000, sample_size=sample_size)
    valid_db = LmReconstructionDatabase("valid", batches_per_epoch=100, sample_size=sample_size)

    anneal = bool(anneal)
    print anneal

    model = make_model(z, sample_size, p, train_db.n_classes, encdec_layers, lstm_size, alpha)
    if not anneal:
        model.anneal = False

    #out = nn.utils.forward(model, train_db, out=model.output(model.input))
    #print out.shape
    #return

    print model.total_params

    name = "lm.charvae.z_%d.len_%d.layers_%d.p_%.2f.alpha_%.2f.lstmsz_%d" % \
           (z, sample_size, encdec_layers, p, alpha, lstm_size)
    if not anneal:
        name += ".noanneal"

    opt = Optimizer(model, train_db, valid_db, Adam(lr),
                    name=name, print_info=True)

    opt.train(100, decay_after=20, lr_decay=0.95)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-z', default=50, type=int)
    parser.add_argument('-lr', default=0.001, type=float)
    parser.add_argument('-p', default=0.0, type=float)
    parser.add_argument('-alpha', default=0.0, type=float)
    parser.add_argument('-sample_size', default=56, type=int)
    parser.add_argument('-encdec_layers', default=2, type=int)
    parser.add_argument('-lstm_size', default=500, type=int)
    parser.add_argument('-anneal', default=1, type=int)
    args = parser.parse_args()
    main(**vars(args))
