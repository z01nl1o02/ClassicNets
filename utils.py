import mxnet as mx
from mxnet import gluon,nd,autograd
import numpy as np


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
        print("epoch {} lr {}".format(epoch,trainer.learning_rate))
        print("\ttrain loss {} {}".format(train_loss / len(train_iter), cls_acc.get()))
        acc = test_net(net, valid_iter, ctx)
        if top_acc < acc:
            print('\ttop valid acc {}'.format(acc))
            net_path = '{}alexnet_top_acc_{}_{:.3f}.params'.format(save_prefix,epoch,top_acc)
            net.save_parameters(net_path)
            top_acc = acc




