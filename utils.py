# -*- coding: utf-8 -*-
import mxnet as mx
from mxnet import gluon,nd,autograd
import numpy as np
import cv2


def show_fcn_mask(ind,Y,out):
    groundtruth = (Y[0,0]).asnumpy() * 255
    out = out[0].asnumpy()
    out = (out[1] > out[0]) * 255
    #print out.shape
    cv2.imwrite("{}_groundtruth.jpg".format(ind),np.uint8(groundtruth))
    cv2.imwrite("{}_test.jpg".format(ind),np.uint8(out))
    #cv2.waitKey(-1)

def test_fcn(net, valid_iter, ctx):
    cls_loss = gluon.loss.SoftmaxCrossEntropyLoss(axis=1)
    cls_acc = mx.metric.Accuracy(name="test acc")
    loss_sum = 0
    for ind,batch in enumerate(valid_iter):
        X,Y = batch
        out = X.as_in_context(ctx)
#        for layer in net:
#            out = layer(out)
        out = net(out)
        out = out.as_in_context(mx.cpu())
        #print Y.shape, out.shape
        cls_acc.update(Y,out)
        loss = cls_loss(out, Y)
        loss_sum += loss.mean().asscalar()
        show_fcn_mask(ind,Y,out)
    print("\ttest loss {} {}".format(loss_sum/len(valid_iter),cls_acc.get()))
    return cls_acc.get_name_value()[0][1]


def train_fcn(net, train_iter, valid_iter, batch_size, trainer, ctx, num_epochs, lr_sch, save_prefix):
    cls_loss = gluon.loss.SoftmaxCrossEntropyLoss(axis=1)
    cls_acc = mx.metric.Accuracy(name="train acc")
    top_acc = 0
    iter_num = 0
    for epoch in range(num_epochs):
        #trainer.set_learning_rate(lr_sch(iter_num))
        train_loss, train_acc = 0, 0
        for batch in train_iter:
            iter_num += 1
            trainer.set_learning_rate(lr_sch(iter_num))
            X,Y = batch
            out = X.as_in_context(ctx)
            #print out.shape
            with autograd.record(True):
#                for layer in net:
#                    out = layer(out)
                out = net(out)
                out = out.as_in_context(mx.cpu())
                loss = cls_loss(out, Y)
            loss.backward()
            train_loss += loss.mean().asscalar()
            trainer.step(batch_size)
            cls_acc.update(Y,out)

            nd.waitall()
        print("epoch {} lr {}".format(epoch,trainer.learning_rate))
        print("\ttrain loss {} {}".format(train_loss / len(train_iter), cls_acc.get()))
        acc = test_fcn(net, valid_iter, ctx)
        if top_acc < acc:
            print('\ttop valid acc {}'.format(acc))
            top_acc = acc
            net_path = '{}top_acc_{}_{:.3f}.params'.format(save_prefix,epoch,top_acc)
            net.save_parameters(net_path)
            




def test_net(net, valid_iter, ctx):
    cls_loss = gluon.loss.SoftmaxCrossEntropyLoss()
    cls_acc = mx.metric.Accuracy(name="test acc")
    loss_sum = 0
    for batch in valid_iter:
        X,Y = batch
        out = X.as_in_context(ctx)
        for layer in net:
            out = layer(out)
        out = out.as_in_context(mx.cpu())
        cls_acc.update(Y,out)
        loss = cls_loss(out, Y)
        loss_sum += loss.mean().asscalar()
    print("\ttest loss {} {}".format(loss_sum/len(valid_iter),cls_acc.get()))
    return cls_acc.get_name_value()[0][1]


def train_net(net, train_iter, valid_iter, batch_size, trainer, ctx, num_epochs, lr_sch, save_prefix):
    cls_loss = gluon.loss.SoftmaxCrossEntropyLoss()
    cls_acc = mx.metric.Accuracy(name="train acc")
    top_acc = 0
    iter_num = 0
    for epoch in range(num_epochs):
        trainer.set_learning_rate(lr_sch(iter_num))
        train_loss, train_acc = 0, 0
        for batch in train_iter:
            iter_num += 1
            X,Y = batch
            out = X.as_in_context(ctx)
            with autograd.record(True):
                for layer in net:
                    out = layer(out)
                out = out.as_in_context(mx.cpu())
                loss = cls_loss(out, Y)
            loss.backward()
            train_loss += loss.mean().asscalar()
            trainer.step(batch_size)
            cls_acc.update(Y,out)

            nd.waitall()
        print("epoch {} lr {}".format(epoch,trainer.learning_rate))
        print("\ttrain loss {} {}".format(train_loss / len(train_iter), cls_acc.get()))
        acc = test_net(net, valid_iter, ctx)
        if top_acc < acc:
            top_acc = acc
            print('\ttop valid acc {}'.format(acc))
            net_path = '{}alexnet_top_acc_{}_{:.3f}.params'.format(save_prefix,epoch,top_acc)
            net.save_parameters(net_path)
            





###########################################################
##rnn
def predict_rnn_gluon(prefix, num_chars, model, vocab_size, ctx, idx_to_char, char_to_idx):
    state = model.begin_state(batch_size=1, ctx=ctx)
    output = [char_to_idx[prefix[0]]]
    for t in range(num_chars + len(prefix) - 1):
        X = nd.array([output[-1]], ctx=ctx).reshape((1,1))
        (Y,state) = model(X,state)
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t+1]])
        else:
            output.append(int(Y.argmax(axis=1).asscalar()))
    return ''.join([idx_to_char[i] for i in output])

import time
import math

def grad_clipping(params, theta, ctx):
    norm = nd.array([0.0],ctx)
    for param in params:
        norm += (param.grad ** 2).sum()
    norm = norm.sqrt().asscalar()
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm
    return


def train_and_predict_rnn_gluon(model,
                                dataset,
                                trainer,
                                loss,
                                ctx,
                                data_iter_func,
                                data_iter_func_type, #random/consecutive
                                num_epochs, num_steps,
                                clipping_theta,
                                batch_size,
                                pred_period,
                                pred_len,
                                prefixes):
   # loss = gluon.loss.SoftmaxCrossEntropyLoss()
   # model.initialize(ctx=ctx, force_reinit=True, init=mx.init.Normal(0.01))
   # trainer = gluon.Trainer(model.collect_params(), "sgd",{'learning_rate':lr, "momentum":0, "wd":0})
    for epoch in range(num_epochs):
        loss_sum, start = 0.0, time.time()
        if data_iter_func_type != "random":
            state = model.begin_state(batch_size=batch_size,ctx=ctx)
        data_iter = data_iter_func(dataset.corpus_indices, batch_size, num_steps, ctx)
        for t,(X,Y) in enumerate(data_iter):
            if data_iter_func_type == "random":
                state = model.begin_state(batch_size=batch_size,ctx=ctx)
            else:
                for s in state:
                    s.detach()
            #?? hidden is not adjusted during training???
            with autograd.record():
                (output,state) = model(X,state)
                y = Y.T.reshape((-1,))
                l = loss(output,y).mean()
            l.backward()
            #grad clip
            params = [p.data() for p in model.collect_params().values()]
            grad_clipping(params, clipping_theta, ctx)
            trainer.step(1) #loss is mean so here step = 1
            loss_sum += l.asscalar()
        if (epoch + 1) % pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec' %(
                epoch + 1, math.exp(loss_sum / (t+1)), time.time() - start
            ))
            for prefix in prefixes:
                print(' -' + predict_rnn_gluon(
                    prefix, pred_len, model, dataset.vocab_size, ctx, dataset.idx_to_char, dataset.char_to_idx

                ))


if 0:
    from datasets.jaychou_lyrics import JAYCHOU_LYRICS
    from networks.rnn import RNNModel
    ctx = mx.gpu()
    lyrics = JAYCHOU_LYRICS(dev_root='datasets/')
    model = RNNModel(lyrics.vocab_size)
    model.initialize(force_reinit=True,ctx=ctx)
    output =  predict_rnn_gluon("分开".decode('utf-8'),10, model, lyrics.vocab_size, ctx, lyrics.idx_to_char, lyrics.char_to_idx)
    print output
