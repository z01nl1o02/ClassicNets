# -*- coding: utf-8 -*-
import mxnet as mx
from mxnet import gluon,nd,autograd
import numpy as np
import cv2
from mxnet import lr_scheduler

import os
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'

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


handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
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
            X,Y = X.as_in_context(ctx), Y.as_in_context(ctx)
            with autograd.record(True):
                out = net(X)
                loss = cls_loss(out, Y)
            loss.backward()
            nd.waitall()
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
        X,Y = X.as_in_context(ctx),Y.as_in_context(ctx)
        out = net(X)
        #out = out.as_in_context(mx.cpu())
        cls_acc.update(Y,out)
        loss = cls_loss(out, Y)
        test_loss.append( loss.sum().asscalar() )
    logger.info("\ttest loss {} {}".format( np.mean(test_loss)/batch_size,cls_acc.get()))
    return cls_acc.get_name_value()[0][1],np.mean(test_loss)/batch_size



def train_net(net, train_iter, valid_iter, batch_size, trainer, ctx, num_epochs, lr_sch, save_prefix):
    logger.info("===================START TRAINING====================")
    if use_mxboard:
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
           # print("iter ",iter_num," start")
            if isinstance(batch,mx.io.DataBatch):
                X,Y = batch.data[0],batch.label[0]
                #total += X.shape[0]
                #print(total)
            else:
                X,Y = batch
            #print(X.shape,Y.shape)
            #print(Y)
            X = X.as_in_context(ctx)
            Y = Y.as_in_context(ctx)
            with autograd.record(True):
                out = net(X)
                #out = out.as_in_context(mx.cpu())                 
                loss = cls_loss(out, Y)        
           # print(out.asnumpy()[0])
           # print('loss = ',loss.sum().asscalar())
            loss.backward()
            train_loss.append( loss.sum().asscalar() )
            trainer.step(batch_size)
            cls_acc.update(Y,out)
            nd.waitall()
            #print("iter ",iter_num," end")
            if use_mxboard:
                if iter_num % 100 == 0:
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


import gluoncv as gcv
def ssd_calc_loss_custom(cls_preds, cls_labels, bbox_preds, bbox2target, bbox_masks):
    loss_func = gcv.loss.SSDMultiBoxLoss()

    return loss_func(cls_preds, bbox_preds, cls_labels, bbox2target)

    


def ssd_calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):

    cls_loss = gluon.loss.SoftmaxCrossEntropyLoss()
    #bbox_loss = gluon.loss.L1Loss()
    bbox_loss = gluon.loss.HuberLoss()
    #print(cls_preds.shape, cls_labels.shape)

    batch_size,anchor_size,cls_num = cls_preds.shape
    cls_preds_ = nd.reshape(cls_preds, (-1,cls_preds.shape[-1]))
    cls_labels_ = nd.reshape(cls_labels, (-1,1))
    #cls_mask = (cls_labels_[:,0] >= 0).reshape( cls_labels_.shape  ) #???? including background?
    cls_mask = (cls_labels_[:, 0] > 0).reshape(cls_labels_.shape)  # ???? including background?

    indices =  nd.array( np.where( cls_mask.asnumpy() > 0)[0], ctx = cls_preds.context )

    cls_preds_valid = nd.take(cls_preds_, indices)
    cls_labels_valid = nd.take(cls_labels_, indices)
    cls = cls_loss(cls_preds_valid, cls_labels_valid)


    bbox_labels = nd.reshape(bbox_labels, (-1,4))
    bbox_masks = nd.sum( nd.reshape(bbox_masks,(-1,4)), axis = -1)
    bbox_preds = nd.reshape(bbox_preds,(-1,4))


    indices =  nd.array( np.where( bbox_masks.asnumpy() > 0)[0], ctx = bbox_preds.context )

    bbox_labels_valid = nd.take(bbox_labels, indices)
    bbox_preds_valid = nd.take(bbox_preds, indices)
    bbox = bbox_loss(bbox_preds_valid, bbox_labels_valid)


    return (cls.mean() + bbox.mean()) * batch_size, cls.mean(), bbox.mean()






 

def predict_ssd(net,X):
    Predict = ssdtool.Prediction(["{}".format(x) for x in range(20)])
    anchors, cls_preds, bbox2target_preds = net(X)
    ids, scores, bboxes = Predict(anchors.as_in_context(mx.cpu()), cls_preds.as_in_context(mx.cpu()), bbox2target_preds.as_in_context(mx.cpu()))

    return nd.concat(ids,scores,bboxes,dim=-1)


import gluoncv as gcv

from tools import ssd as ssdtool
from tqdm import tqdm

def test_ssd_custom(net, valid_iter, ctx):
    mAP = gcv.utils.metrics.voc_detection.VOC07MApMetric(iou_thresh=0.5, class_names=('aeroplane', 'bicycle', 'bird',
                         'boat', 'bottle', 'bus', 'car', 'cat', 'chair','cow', 'diningtable', 'dog', 'horse',
                          'motorbike', 'person', 'pottedplant','sheep', 'sofa', 'train', 'tvmonitor'))
    start = time.time()
    loss_cls_hist, loss_bbox_hist = [], []
    loss_hist = []
    Predict = ssdtool.Prediction(["{}".format(x) for x in range(20)])
    AssignTargetFor = ssdtool.AssginTarget()
    for batch in tqdm(valid_iter):
        X = batch[0].as_in_context(ctx)
        Y = batch[1].as_in_context(ctx)
        anchors, cls_preds, bbox2target_preds = net(X)
        cls_labels, bbox2target, bbox_masks = AssignTargetFor(anchors, cls_preds, bbox2target_preds, Y)

        l, l_cls, l_bbox = ssd_calc_loss_custom(cls_preds, cls_labels, bbox2target_preds, bbox2target,
                                                bbox_masks)
        loss_hist.append( nd.concatenate(l).mean().asnumpy()[0] )
        loss_bbox_hist.append(  nd.concatenate(l_bbox).mean().asnumpy()[0] )
        loss_cls_hist.append(  nd.concatenate(l_cls).mean().asnumpy()[0] )
        ids, scores, bboxes = Predict(anchors.as_in_context(mx.cpu()), cls_preds.as_in_context(mx.cpu()), bbox2target_preds.as_in_context(mx.cpu()))
        gt_bboxes = nd.slice_axis(batch[1],axis=-1, begin=1,end=None)
        gt_lables = nd.slice_axis(batch[1],axis=-1, begin=0,end=1)
        mAP.update(pred_bboxes=bboxes, pred_labels=ids,pred_scores=scores,gt_bboxes=gt_bboxes, gt_labels = gt_lables)
    loss = np.asarray(loss_hist).mean()
    loss_bbox = np.mean(loss_bbox_hist)
    loss_cls = np.mean(loss_cls_hist)
    logger.info('\t test class loss %.5e, bbox loss %.5e, loss %.5e, time %.1f sec' % (
        loss_cls, loss_bbox, loss, time.time() - start))
    names, values = mAP.get()
    for name,value in zip(names,values):
        print(name, value)
    return values[-1]


def train_ssd_custom(net, train_iter, valid_iter, batch_size, trainer, ctx, num_epochs, lr_sch, save_prefix):
    logger.info("===================START TRAINING====================")
    start = time.time()
    AssignTargetFor = ssdtool.AssginTarget()
   # test_ssd_custom(net, valid_iter, ctx)


    last_map = 0
    for epoch in range(num_epochs):
        #acc_hist, mae_hist = [],[]
        loss_cls_hist, loss_bbox_hist = [], []
        loss_hist = []
        trainer.set_learning_rate(lr_sch(epoch))
        for batch in train_iter:
            X = batch[0].as_in_context(ctx)
            Y = batch[1].as_in_context(ctx)
            with autograd.record():
                # 生成多尺度的锚框，为每个锚框预测类别和偏移量
                anchors, cls_preds, bbox_preds = net(X)
                cls_labels, bbox_labels, bbox_masks = AssignTargetFor(anchors, cls_preds, bbox_preds, Y)
               # bbox_labels, bbox_masks, cls_labels = contrib.nd.MultiBoxTarget(
                #    anchors, Y, cls_preds.transpose((0, 2, 1)), negative_mining_ratio = 3.0)
                # 根据类别和偏移量的预测和标注值计算损失函数
                l,l_cls, l_bbox = ssd_calc_loss_custom(cls_preds, cls_labels, bbox_preds, bbox_labels,
                              bbox_masks)
            #nd.concatenate(l).backward()
            autograd.backward(l)
            trainer.step(1)
            nd.waitall()
            loss_hist.append(nd.concatenate(l).mean().asnumpy()[0])
            loss_bbox_hist.append(nd.concatenate(l_bbox).mean().asnumpy()[0])
            loss_cls_hist.append(nd.concatenate(l_cls).mean().asnumpy()[0])


        if (epoch + 1)%1 == 0:
            loss = np.asarray(loss_hist).mean()
            loss_bbox = np.mean(loss_bbox_hist)
            loss_cls = np.mean(loss_cls_hist)
            logger.info('epoch %2d, class loss %.5e, bbox loss %.5e, loss %.5e, lr %.5e time %.1f sec' % (
                epoch + 1, loss_cls,loss_bbox, loss, trainer.learning_rate, time.time() - start))
            start = time.time() #restart

        if (epoch + 1)%2 == 0:
            mAP = test_ssd_custom(net,valid_iter,ctx)
            if mAP > last_map:
                net.save_parameters("{}_epoch{}_map{}.params".format(save_prefix,epoch,mAP))
                last_map = mAP


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


class DiceLoss(mx.gluon.Block):
    def __init__(self,smooth=0.01, from_logists=False, axis=-1, sparse_label = True):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.from_logists = from_logists
        self.axis = axis
        self.sparse_label = sparse_label
        return
    def forward(self, pred, label):
        num = pred.shape[0]
        if not self.from_logists:
            pred = nd.softmax(pred, self.axis)
        if self.sparse_label:
            with autograd.pause():
                label_dense = nd.zeros_like(pred)
                for l in range(label_dense.shape[1]):
                    label_dense[:,l,:] = (label == l) * 1.0
                label = label_dense
        pred, label = nd.reshape(pred, (num,-1)), nd.reshape(label,(num,-1))
        union = pred.sum() + label.sum()
        inter = (pred * label).sum()
        return 1 - (2 * inter + self.smooth) / (self.smooth + union)





if 0:
    label = nd.zeros((10,))
    label[3] = 2
    pred = nd.random.uniform(0,1,shape = (10,4))
    loss = LabelSmoothSCE()
    loss_value = loss(pred,label)
    print('loss value: ', loss_value)


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
