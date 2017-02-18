####  A Hybrid Convolutional Variational Autoencoder for Text Generation.

Theano code for experiments in the paper [A Hybrid Convolutional Variational Autoencoder for Text Generation](https://arxiv.org/abs/1702.02390).

#### Preparation

First, run makedata.sh. This will download the ptb dataset, split, and preprocess it.

#### PTB Experiments

Files prefixed with ''lm_'' contain experiments on the ptb dataset. We provide scripts for training of non-VAE, baseline LSTM VAE, and our models and a script to greedily sample from a trained model. ''defs'' subfolder contains definitions of grid searches we have used to generate data for figures and tables in the paper. Running one search is done by:
```bash
python -u nn/scripts/grid_search.py -grid defs/gridname.json
```
To train our model on samples 60 characters long with alpha=0.2 run:
```bash
python -u lm_vae_lstm.py -alpha 0.2 -sample_size 60
```

#### Twitter Experiments

Code for these experiments is in files starting with ''twitter_''. We do not release the dataset we have used to train our model, but provide both a script to train one and [a pretrained model](https://dl.dropboxusercontent.com/u/60972596/pretrained.tar). To use the script on custom data, create a file ''data/tweets.txt'' containing one data sample per line. By default, the first 10k samples will be used for validation and everything else for training, but no more than ~1M samples. In addition, it will only use tweets with up to 128 characters. This is done only for convenience when down- and upsampling. Training on tweets with up to 140 characters will require a little bit of care when handling spatial dimension.

#### License

MIT
