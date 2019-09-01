# coding=utf-8
import mxnet as mx
import os
from datasets.jaychou_lyrics import JAYCHOU_LYRICS,data_iter_consecutive,data_iter_random
from networks.rnn import RNNModel
from mxnet import gluon
from mxnet.gluon import Trainer
from utils import train_and_predict_rnn_gluon,predict_rnn_gluon

data_iter_func_type = "random"
batch_size = 32
ctx = mx.gpu()
num_steps = 6
num_max_vocab = 10000

num_epoch = 1000

num_hidden = 256
base_lr = 100 ##????
clipping_theta = 0.01

pred_period = 50 #frequence to run prediction
pred_len = 50 #len to prediction
prefixes = ['分开','不分开']

prefixes = map(lambda x: x.decode('utf-8'), prefixes) #encoding problem

dataset_lyrics = JAYCHOU_LYRICS(root = os.path.join(os.getcwd(),'datasets'), num_max = num_max_vocab)

if data_iter_func_type == "random":
    data_iter_func = data_iter_random
else:
    data_iter_func = data_iter_consecutive

model = RNNModel(vocab_size=dataset_lyrics.vocab_size,num_hidden = num_hidden)
model.initialize(mx.init.Uniform(0.01), ctx=ctx,force_reinit=True)


trainer = Trainer(model.collect_params(), "sgd",{'learning_rate':base_lr, "momentum":0, "wd":0})
loss = gluon.loss.SoftmaxCrossEntropyLoss()


train_and_predict_rnn_gluon(model,dataset_lyrics,trainer,loss,ctx, data_iter_func,
                            data_iter_func_type, num_epoch, num_steps, clipping_theta,
                            batch_size, pred_period, pred_len, prefixes)






