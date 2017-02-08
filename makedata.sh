mkdir data
mkdir data/ptb

wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
tar --extract --file=simple-examples.tgz ./simple-examples/data/ptb.train.txt
tar --extract --file=simple-examples.tgz ./simple-examples/data/ptb.test.txt
tar --extract --file=simple-examples.tgz ./simple-examples/data/ptb.valid.txt

mv simple-examples/data/ptb.train.txt ./data/ptb/.
mv simple-examples/data/ptb.test.txt ./data/ptb/.
mv simple-examples/data/ptb.valid.txt ./data/ptb/.

rm -r simple-examples*

python -u -m nn.scripts.prepare_char_lm
