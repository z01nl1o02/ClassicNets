# -*- coding: utf-8 -*-
import mxnet as mx
from mxnet import gluon,nd,autograd
import numpy as np
import cv2,os,pdb,time
from mxnet import lr_scheduler

use_mxboard = False

if use_mxboard:
    from mxboard import SummaryWriter
from mxnet import contrib

import logging
logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)
handler = logging.FileHandler("log.txt")
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class MIOU:
    """
    Computes MIOU
    Parameters
    ----------
    axis : int, default=1
        The axis that represents classes
    name : str
        Name of this metric instance for display.
    output_names : list of str, or None
        Name of predictions that should be used when updating with update_dict.
        By default include all predictions.
    label_names : list of str, or None
        Name of labels that should be used when updating with update_dict.
        By default include all labels.
    """
    def __init__(self, class_names, axis=1, name='mIOU'):
        self.axis = axis
        self.n = len(class_names)
        self.classes = class_names
        self.name = name
        self.hist = np.bincount([0], minlength=self.n ** 2).reshape(self.n, self.n)

    def _get_hist(self,labels, preds):
        k = (labels >= 0) & (labels < self.n)
        return np.bincount(self.n * labels[k].astype(int) + preds[k], minlength=self.n ** 2).reshape(self.n, self.n)

    def reset(self):
        self.hist = np.bincount([0], minlength=self.n ** 2).reshape(self.n, self.n)
        return

    def get(self):
        miou = np.diag(self.hist) * 1.0 / (self.hist.sum(1) + self.hist.sum(0) - np.diag(self.hist))
        return (self.name, np.nanmean(miou))

    def detail(self):
        lines = []
        miou = np.diag(self.hist) * 1.0 / (self.hist.sum(1) + self.hist.sum(0) - np.diag(self.hist))
        for k, cls in enumerate(self.classes):
            lines.append('({}:{:.3f})'.format(cls,miou[k]))
        return ' '.join(lines)

    def update(self, labels, preds):
        """Updates the internal evaluation result.

        Parameters
        ----------
        labels : list of `NDArray`
            The labels of the data with class indices as values, one per sample.

        preds : list of `NDArray`
            Prediction values for samples. Each prediction value can either be the class index,
            or a vector of likelihoods for all classes.
        """
        labels, preds = mx.metric.check_label_shapes(labels, preds, True)

        for label, pred_label in zip(labels, preds):
            if pred_label.shape != label.shape:
                pred_label = mx.nd.argmax(pred_label, axis=self.axis)
            pred_label = pred_label.asnumpy().astype('int32')
            label = label.asnumpy().astype('int32')
            # flatten before checking shapes to avoid shape miss match
            label = label.flat
            pred_label = pred_label.flat

            mx.metric.check_label_shapes(label, pred_label)
            self.hist += self._get_hist(label, pred_label)
        return
    def get_name_value(self):
        """Returns zipped name and value pairs.

        Returns
        -------
        list of tuples
            A (name, value) tuple list.
        """
        name, value = self.get()
        if not isinstance(name, list):
            name = [name]
        if not isinstance(value, list):
            value = [value]
        return list(zip(name, value))


def show_seg_mask(net,ind,Y,out):
    if os.path.exists("debug"):
        groundtruth = (Y[0]).asnumpy() * 10
        out = out[0].asnumpy()
        out = np.argmax(out,axis=0) * 10
        cv2.imwrite("debug/{}_groundtruth.jpg".format(ind),np.uint8(groundtruth))
        cv2.imwrite("debug/{}_test.jpg".format(ind),np.uint8(out))
        if 0:
            for name in net.collect_params('.*weight'):
                w = net.collect_params()[name]
                print(name, w.data().asnumpy().mean(), w.data().asnumpy().std())


def test_seg(net, valid_iter, ctx, debug_show, cls_acc = None, cls_loss = None):
    if cls_loss is None:
        cls_loss = gluon.loss.SoftmaxCrossEntropyLoss(axis=1)
    if cls_acc is None:
        cls_acc = mx.metric.Accuracy(name="test acc")
    cls_acc.reset()
    loss_sum = 0
    for ind,batch in enumerate(valid_iter):
        X,Y = batch
        out = X.as_in_context(ctx)
#        for layer in net:
#            out = layer(out)
        #with autograd.predict_mode():
        out = net(out)
        out = out.as_in_context(mx.cpu())
       # print Y.shape, out.shape
       # pdb.set_trace()
        cls_acc.update(Y,out)
        loss = cls_loss(out, Y)
        loss_sum += loss.mean().asscalar()
        if debug_show:
            show_seg_mask(net,ind,Y,out)
    logger.info("\ttest loss {} {}".format(loss_sum/len(valid_iter),cls_acc.get()))
    logger.info("\t\t{}".format(cls_acc.detail()))
    return cls_acc.get_name_value()[0][1]


def train_seg(net, train_iter, valid_iter, batch_size, trainer, ctx, num_epochs, lr_sch, cls_acc = None, cls_loss = None, save_prefix = "./"):
    if cls_loss is None:
        cls_loss = gluon.loss.SoftmaxCrossEntropyLoss(axis=1)
    if cls_acc is None:
        cls_acc = mx.metric.Accuracy(name="train acc")
    top_acc = 0
    iter_num = 0
   
    for epoch in range(num_epochs):
        #trainer.set_learning_rate(lr_sch(iter_num))
        cls_acc.reset()
        train_loss, train_acc = 0, 0
        for batch in train_iter:
            iter_num += 1
            trainer.set_learning_rate(lr_sch(iter_num))
            X,Y = batch
            out = X.as_in_context(ctx)
            with autograd.record(True):
                out = net(out)
                out = out.as_in_context(mx.cpu())
                #print(out.shape,Y.shape)
                loss = cls_loss(out, Y)
            loss.backward()
            nd.waitall()
            #print loss
            train_loss += loss.mean().asscalar()
            trainer.step(batch_size)
            cls_acc.update(Y,out)
        logger.info("epoch {} lr {}".format(epoch,trainer.learning_rate))
        logger.info("\ttrain loss {} {}".format(train_loss / len(train_iter), cls_acc.get()))
        
        if (epoch % 10) == 0:    
            acc = test_seg(net, valid_iter, ctx, debug_show = True, cls_acc = cls_acc,cls_loss = cls_loss)
            net_path = '{}last_model.params'.format(save_prefix)
            net.save_parameters(net_path)
            if top_acc < acc:
                logger.info('\ttop valid acc {}'.format(acc))
                top_acc = acc
                net_path = '{}top_acc_{}_{:.5f}.params'.format(save_prefix,epoch,top_acc)
                net.save_parameters(net_path)
            




def test_net(net, valid_iter, ctx):
    cls_loss = gluon.loss.SoftmaxCrossEntropyLoss()
    cls_acc = mx.metric.Accuracy(name="test acc")
    test_loss = []
    batch_size = 1
    if isinstance(valid_iter,mx.io.MXDataIter):
        valid_iter.reset()
        
    for batch in valid_iter:
        if isinstance(batch,mx.io.DataBatch):
            X,Y = batch.data[0],batch.label[0]
            #print(X.shape,Y.shape)
        else:
            X,Y = batch
        batch_size = X.shape[0]
        out = X.as_in_context(ctx)
        out = net(out)
        out = out.as_in_context(mx.cpu())
        cls_acc.update(Y,out)
        loss = cls_loss(out, Y)
        test_loss.append( loss.sum().asscalar() )
    logger.info("\ttest loss {} {}".format( np.mean(test_loss)/batch_size,cls_acc.get()))
    return cls_acc.get_name_value()[0][1],np.mean(test_loss)/batch_size



def train_net(net, train_iter, valid_iter, batch_size, trainer, ctx, num_epochs, lr_sch, save_prefix):
    logger.info("===================START TRAINING====================")
    sw = SummaryWriter(logdir='logs', flush_secs=5)
    cls_loss = gluon.loss.SoftmaxCrossEntropyLoss()
    cls_acc = mx.metric.Accuracy(name="train acc")
    top_acc = 0
    iter_num = 0
    #test_acc,test_loss = test_net(net, valid_iter, ctx)
    #sw.add_graph(net) #only hybrid block supported
    param_names = net.collect_params().keys()
    for epoch in range(num_epochs):
        train_loss = []
        t0 = time.time()
        if isinstance(train_iter,mx.io.MXDataIter):
            train_iter.reset()
        total = 0
        trainer.set_learning_rate(lr_sch(epoch))
        for batch in train_iter:
            iter_num += 1
            if isinstance(batch,mx.io.DataBatch):
                X,Y = batch.data[0],batch.label[0]
                #total += X.shape[0]
                #print(total)
            else:
                X,Y = batch
            #print(X.shape,Y.shape)
            #print(Y)
            out = X.as_in_context(ctx)
            with autograd.record(True):
                out = net(out)
                out = out.as_in_context(mx.cpu())
                loss = cls_loss(out, Y)
            loss.backward()
            train_loss.append( loss.sum().asscalar() )
            trainer.step(batch_size)
            cls_acc.update(Y,out)
            nd.waitall()
            if use_mxboard:
                sw.add_scalar(tag='train_loss', value=loss.mean().asscalar(), global_step=iter_num)
                sw.add_scalar(tag='train_acc', value=cls_acc.get(), global_step=iter_num)
                if iter_num % 100 == 0:
                    for name in net.collect_params():
                        param = net.collect_params()[name]
                        if param.grad_req != "null":
                            sw.add_histogram(tag=name, values=param.grad(), global_step=iter_num, bins=1000)

            
            
        logger.info("epoch {} lr {} {}sec".format(epoch,trainer.learning_rate, time.time() - t0))
        train_loss, train_acc = np.mean(train_loss) / batch_size, cls_acc.get()
        logger.info("\ttrain loss {} {}".format(train_loss, train_acc))
        if epoch > 0 and (epoch % 10) == 0:
            test_acc,test_loss = test_net(net, valid_iter, ctx)
            if use_mxboard:
                sw.add_scalar(tag='test_acc', value=test_acc, global_step=epoch)
                sw.add_scalar(tag='test_loss', value=test_loss, global_step=epoch)
            if top_acc < test_acc:
                top_acc = test_acc
                logger.info('\ttop valid acc {}'.format(test_acc))
                if isinstance(net, mx.gluon.nn.HybridSequential) or isinstance(net, mx.gluon.nn.HybridBlock):
                    pf = '{}_{:.3f}.params'.format(save_prefix,top_acc)
                    net.export(pf,epoch)
                else:
                    net_path = '{}top_acc_{}_{:.3f}.params'.format(save_prefix,epoch,top_acc)
                    net.save_parameters(net_path)
                
    if use_mxboard:
        sw.close()

##############################################################
##ssd


def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
    cls_loss = gluon.loss.SoftmaxCrossEntropyLoss()
    bbox_loss = gluon.loss.L1Loss()
    cls = cls_loss(cls_preds, cls_labels)
    bbox = bbox_loss(bbox_preds * bbox_masks, bbox_labels * bbox_masks)
    return cls + bbox

def cls_eval(cls_preds, cls_labels):
    # 由于类别预测结果放在最后一维，argmax需要指定最后一维
    return (cls_preds.argmax(axis=-1) == cls_labels).sum().asscalar()

def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
   # print (bbox_labels*bbox_masks)
  #  print (bbox_preds*bbox_masks).sum()
    return ((bbox_labels - bbox_preds) * bbox_masks).abs().sum().asscalar()
  
import pdb
def predict_ssd(net,X):
    anchors, cls_preds, bbox_preds = net(X)
    cls_probs = cls_preds.softmax().transpose((0, 2, 1))
    output = contrib.nd.MultiBoxDetection(cls_probs, bbox_preds, anchors)
    idx = [i for i, row in enumerate(output[0]) if row[0].asscalar() != -1]
    if len(idx) < 1:
        return mx.nd.zeros((1,output.shape[-1])) - 1
    return output[0, idx]  
    
    
def test_net(net, valid_iter, ctx):
    start = time.time()
    acc_sum, mae_sum, n, m = 0.0, 0.0, 0, 0
    loss_hist = []
    for batch in valid_iter:        
        X = batch[0].as_in_context(ctx)
        Y = batch[1].as_in_context(ctx)        
        anchors, cls_preds, bbox_preds = net(X)
        # 为每个锚框标注类别和偏移量
        #pdb.set_trace()
        bbox_labels, bbox_masks, cls_labels = contrib.nd.MultiBoxTarget(
            anchors, Y, cls_preds.transpose((0, 2, 1)))
        # 根据类别和偏移量的预测和标注值计算损失函数
        l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels,
                      bbox_masks)
        loss_hist.append( l.asnumpy()[0] / X.shape[0] )
        acc_sum += cls_eval(cls_preds, cls_labels)
        n += cls_labels.size
        mae_sum += bbox_eval(bbox_preds, bbox_labels, bbox_masks)
        m += bbox_labels.size
    loss = np.asarray(loss_hist).mean()
    logger.info('\t test class err %.5e, bbox mae %.5e, loss %.5e, time %.1f sec' % ( 
        1 - acc_sum / n, mae_sum / m, loss, time.time() - start))
    return
    
    
def train_ssd(net, train_iter, valid_iter, batch_size, trainer, ctx, num_epochs, lr_sch, save_prefix):
    logger.info("===================START TRAINING====================")
    start = time.time()
    for epoch in range(num_epochs):
        acc_sum, mae_sum, n, m = 0.0, 0.0, 0, 0
        loss_hist = []
        trainer.set_learning_rate(lr_sch(epoch))
        for batch in train_iter:        
            X = batch[0].as_in_context(ctx)
            Y = batch[1].as_in_context(ctx)
            #print(X)
            #print(X.shape,Y.shape)
            with autograd.record():
                # 生成多尺度的锚框，为每个锚框预测类别和偏移量
                anchors, cls_preds, bbox_preds = net(X)
                # 为每个锚框标注类别和偏移量
                #pdb.set_trace()
                bbox_labels, bbox_masks, cls_labels = contrib.nd.MultiBoxTarget(
                    anchors, Y, cls_preds.transpose((0, 2, 1)))
                # 根据类别和偏移量的预测和标注值计算损失函数
                l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels,
                              bbox_masks)
            l.backward()
            trainer.step(batch_size)
            #nd.waitall()
            loss_hist.append( l.asnumpy()[0] / batch_size )
            acc_sum += cls_eval(cls_preds, cls_labels)
            n += cls_labels.size
            mae_sum += bbox_eval(bbox_preds, bbox_labels, bbox_masks)
            m += bbox_labels.size

	if (epoch + 1)%2 == 0:
            loss = np.asarray(loss_hist).mean()
            logger.info('epoch %2d, class err %.5e, bbox mae %.5e, loss %.5e, lr %.5e time %.1f sec' % (
                epoch + 1, 1 - acc_sum / n, mae_sum / m, loss, trainer.learning_rate, time.time() - start))
            start = time.time() #restart    

        if (epoch + 1) % 50 == 0:
            test_net(net,valid_iter,ctx)
            net.save_parameters("{}_epoch{}.params".format(save_prefix,epoch))   

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



class CycleScheduler(lr_scheduler.LRScheduler):
    def __init__(self,updates_one_cycle, min_lr, max_lr):
        super(CycleScheduler,self).__init__()
        self.updates_one_cycle = np.float32(updates_one_cycle)
        self.min_lr = min_lr
        self.max_lr = max_lr
        return
    def __call__(self,update):
        update = update % self.updates_one_cycle
        lr = self.min_lr + (self.max_lr - self.min_lr) * update / self.updates_one_cycle
        return lr



class FocusLoss(mx.gluon.loss.Loss):
    #copy from mx.gluon.loss.softmaxloss
    def __init__(self, alpha = 1.0, gamma=1.0, axis=-1, sparse_label=True, from_logits=False,
                 batch_axis=0, **kwargs):
        super(FocusLoss, self).__init__(None, batch_axis, **kwargs)
        self._axis = axis
        self._gamma = gamma
        self._alpha = alpha
        self._sparse_label = sparse_label
        self._from_logits = from_logits

    def hybrid_forward(self, F, pred, label):
        if not self._from_logits:
            pred = F.softmax(pred,self._axis)
            adjW = self._alpha * ((1-pred)**self._gamma) #focus loss
            pred = adjW * F.log(pred)
        else:
            return 0
        if self._sparse_label:
            loss = -F.pick(pred, label, axis=self._axis, keepdims=True)
        else:
            label = mx.gluon.loss.reshape_like(F, label, pred)
            loss = -F.sum(pred*label, axis=self._axis, keepdims=True)
        loss = mx.gluon.loss._apply_weighting(F, loss, self._weight, None)
        loss = F.mean(loss, axis=self._batch_axis, exclude=True)
#        print 'focus loss: ',loss
        return loss



class WeightCELoss(mx.gluon.loss.Loss):
    #copy from mx.gluon.loss.softmaxloss
    def __init__(self, weight_classes, axis=-1, sparse_label=True, from_logits=False,
                 batch_axis=0, **kwargs):
        super(WeightCELoss, self).__init__(None, batch_axis, **kwargs)
        self._axis = axis
        self._weight_classes = weight_classes
        self._sparse_label = sparse_label
        self._from_logits = from_logits

    def hybrid_forward(self, F, pred, label):
        if not self._from_logits:
            pred = F.softmax(pred,self._axis)
            pred = F.log(pred)
        else:
            return 0
        if self._sparse_label:
            loss = -F.pick(pred, label, axis=self._axis, keepdims=True)
        else:
            label = mx.gluon.loss.reshape_like(F, label, pred)
            loss = -F.sum(pred*label, axis=self._axis, keepdims=True)

        sample_weight = None
        if self._weight_classes:
            label_cpu = label.asnumpy()
            class_weight = np.asarray(self._weight_classes)
            sample_weight = np.choose(label_cpu,class_weight)
            sample_weight = nd.array(sample_weight)
            if (np.isnan(sample_weight.asnumpy()).sum() > 0):
                print("nan sample_weight")
        loss = mx.gluon.loss._apply_weighting(F, loss, self._weight, sample_weight=sample_weight)
        loss = F.mean(loss, axis=self._batch_axis, exclude=True)
#        print 'focus loss: ',loss
        return loss


class SpatialDropout2D(mx.gluon.Block):
    def __init__(self, p):
        super(SpatialDropout2D, self).__init__()
        self.p = p

    def forward(self, x):
        if not autograd.is_training():
            return x
        mask_shape = x.shape[:2] + (1, 1)
        mask = nd.random.multinomial(nd.array([self.p, 1 - self.p],ctx = x.context),
                                     shape=mask_shape).astype('float32')
        return (x * mask) / (1 - self.p)


if 0:
    x = nd.ones((1, 10, 3, 3))
    x.attach_grad()
    net = SpatialDropout2D(0.5)
    with autograd.record():
        out = net(x)
    out.backward()
    print(x)
    print(x.grad)

if 0:
    from datasets.jaychou_lyrics import JAYCHOU_LYRICS
    from networks.rnn import RNNModel
    ctx = mx.gpu()
    lyrics = JAYCHOU_LYRICS(dev_root='datasets/')
    model = RNNModel(lyrics.vocab_size)
    model.initialize(force_reinit=True,ctx=ctx)
    output =  predict_rnn_gluon("分开".decode('utf-8'),10, model, lyrics.vocab_size, ctx, lyrics.idx_to_char, lyrics.char_to_idx)
    print(output)
