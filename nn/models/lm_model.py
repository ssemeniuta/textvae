import theano.tensor as T
from base_model import BaseModel


class LMModel(BaseModel):

    def __init__(self, layers):
        super(LMModel, self).__init__(layers)
        self.input = T.imatrix()
        self.target = T.imatrix()

    @property
    def costs(self):
        p = self.output(self.input)
        cost = T.nnet.categorical_crossentropy(p, self.target.flatten()).mean()
        pp = T.exp(cost)
        return [cost, pp]
