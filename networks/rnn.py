import math
from mxnet import gluon,init,nd
from mxnet.gluon import loss as gloss,nn,rnn
import time


class RNNModel(nn.Block):
    def __init__(self,vocab_size, num_hidden,**kwargs):
        super(RNNModel,self).__init__(**kwargs)
        self.rnn = rnn.RNN(num_hidden)
        self.rnn.initialize()
        self.vocab_size = vocab_size
        self.dense = nn.Dense(vocab_size) #output to observation
    def forward(self, inputs, state):
        X = nd.one_hot(inputs.T, self.vocab_size)
        Y,state = self.rnn(X,state)
        output = self.dense(Y.reshape((-1,Y.shape[-1])))
        return output, state
    def begin_state(self,*args,**kwargs):
        return self.rnn.begin_state(*args, **kwargs)
