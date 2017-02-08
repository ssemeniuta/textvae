import argparse
import numpy
import theano
import theano.tensor as T
from databases.lm_reconstruction_database import LmReconstructionDatabase

from vae import Sampler, Dropword,  LMReconstructionModel
from nn.layers1d import LayoutCNNToRNN, LayoutRNNToCNN, Convolution1d, Deconvolution1D
from nn.layers import Linear, Embed, Flatten, Reshape, SoftMax, Dropout, OneHot
from nn.activations import ReLU, Tanh, Gated
from nn.optimizer import Optimizer
from nn.updates import Adam
from nn.rnns import LNLSTM
from nn.containers import Sequential, Parallel
from nn.normalization import BatchNormalization
from nn.clipping import MaxNorm
import nn.utils


class ConditionalDecoderLNLSTM(object):

    def __init__(self, input_size, image_feature_size, layer_size, p=0.0, name="", steps=16):
        self.steps = steps
        self.lstm = LNLSTM(input_size, layer_size, name=name, p=p)
        self.init = Sequential([
            Linear(image_feature_size, 2*layer_size, name=name+"_init"),
            Tanh()
        ])
        self.params = self.init.params + self.lstm.params

    def __call__(self, x):
        z, w = x
        init = self.init(z)
        h_init = init[:, :self.lstm.layer_size]
        c_init = init[:, self.lstm.layer_size:]

        [c, h], self.updates = theano.scan(self.lstm.step, w, outputs_info=[c_init, h_init], n_steps=self.steps)

        return h


def make_model(z, net, sample_size, p, n_classes):
    if net == "conv":
        assert sample_size % 4 == 0
        layers = [
            OneHot(n_classes),
            LayoutRNNToCNN(),
            Convolution1d(3, 128, n_classes, pad=1, stride=2, causal=False, name="conv1"),
            BatchNormalization(128, name="bn1"),
            ReLU(),
            Convolution1d(3, 256, 128, pad=1, stride=2, causal=False, name="conv2"),
            BatchNormalization(256, name="bn2"),
            ReLU(),
            Flatten(),
            Linear(sample_size / 4 * 256, z * 2, name="fc_encode"),
            Sampler(z),
            Linear(z, sample_size / 4 * 256, name="fc_decode"),
            ReLU(),
            Reshape((-1, 256, sample_size / 4, 1)),
            Deconvolution1D(256, 128, 3, pad=1, stride=2, name="deconv2"),
            BatchNormalization(128, name="deconv_bn2"),
            ReLU(),
            Deconvolution1D(128, 200, 3, pad=1, stride=2, name="deconv1"),
            BatchNormalization(200, name="deconv_bn1"),
            ReLU(),
            LayoutCNNToRNN(),
            Linear(200, n_classes, name="classifier"),
            SoftMax()
        ]
    elif net == "rnn":
        start_word = n_classes
        dummy_word = n_classes + 1
        layers = [
            Parallel([
                [
                    OneHot(n_classes),
                    LNLSTM(n_classes, 500, name="enc"),
                    lambda x: x[-1],
                    Linear(500, z * 2, name="encoder_fc"),
                    Sampler(z),
                ],
                [
                    Dropword(p, dummy_word=dummy_word),
                    lambda x: T.concatenate([T.ones((1, x.shape[1]), dtype='int32') * start_word, x], axis=0),
                    lambda x: x[:-1],
                    OneHot(n_classes + 2),
                ]
            ]),
            ConditionalDecoderLNLSTM(n_classes + 2, z, 500, name="dec", steps=sample_size),
            Linear(500, n_classes, name="classifier"),
            SoftMax()
        ]
    else:
        raise Exception("unknown net %s" % net)

    model = LMReconstructionModel(layers, aux_loss=False)

    return model


def main(z, lr, net, sample_size, p, clip, nokl):
    train_db = LmReconstructionDatabase("train", batches_per_epoch=1000, sample_size=sample_size)
    valid_db = LmReconstructionDatabase("valid", batches_per_epoch=100, sample_size=sample_size)

    model = make_model(z, net, sample_size, p, train_db.n_classes)
    if nokl:
        model.anneal_start = 1e20
        model.anneal_end = 1e21

    #out = nn.utils.forward(model, train_db, out=model.output(model.input))
    #print out.shape
    #return

    print model.total_params

    if net == "conv":
        print "not using clipping for conv model"
        clip = 0.0

    name = "vae.%d.%s.%d.%.2f.clip_%d.lr_%.4f" % (z, net, sample_size, p, clip, lr)

    opt = Optimizer(model, train_db, valid_db, Adam(lr), grad_clip=MaxNorm(clip),
                    name=name, print_info=True)

    opt.train(100)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-z', default=100, type=int)
    parser.add_argument('-clip', default=100, type=int)
    parser.add_argument('-lr', default=0.001, type=float)
    parser.add_argument('-p', default=0.0, type=float)
    parser.add_argument('-sample_size', default=52, type=int)
    parser.add_argument('-net', default="rnn")
    parser.add_argument('-nokl', default=False, type=bool)
    args = parser.parse_args()
    main(**vars(args))
