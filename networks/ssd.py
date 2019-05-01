#coding=utf-8
from mxnet import contrib, image, nd,gluon,autograd, init
from mxnet.gluon import loss as gloss, nn
import mxnet as mx
import numpy as np


def conv_layer(out_ch, ks, padding,stride):
    layer = nn.Sequential()
    layer.add(
        nn.Conv2D(out_ch,ks,strides=stride,padding=padding),
        nn.BatchNorm(),
        nn.Activation("relu")
    )
    return layer

def down_sample_blk(num_channels):
    blk = nn.Sequential()
    for _ in range(2):
        blk.add( conv_layer(num_channels,ks=3,padding=1,stride=1) )
    blk.add(nn.MaxPool2D(2))
    return blk


class INCEPTION_BLOCK(nn.Block):
    def __init__(self,channels, **kwargs):
        super(INCEPTION_BLOCK, self).__init__(**kwargs)
        self.p1_1 = nn.Conv2D(channels, kernel_size=1, activation="relu")

        self.p2_1 = nn.Conv2D(channels//2,kernel_size=1, activation="relu")
        self.p2_2 = nn.Conv2D(channels,kernel_size=3, padding=1, activation="relu")

        self.p3_1 = nn.Conv2D(channels//4, kernel_size=1,activation="relu")
        self.p3_2 = nn.Conv2D(channels,kernel_size=5, padding=2, activation="relu")

    def forward(self, x):
        p1 = self.p1_1(x)
        p2 = self.p2_2(self.p2_1(x))
        p3 = self.p3_2(self.p3_1(x))
        return nd.concat(p1,p2,p3,dim=1)
    
    
#类别预测层，这里用一个卷积层实现
#其输入输出层的w/h必须一致
#输出通道数C等于(anchor个数)x(类别数+1)，这里类别不包括背景，+1就是背景类
def cls_predictor(num_anchors, num_classes):
    return nn.Conv2D(num_anchors * (num_classes + 1), kernel_size=3,
                     padding=1)


#边框预测层，也用一个卷积层实现
#输入输出的w/h必须一致
#输出通道数C等于4x(anchor个数)
def bbox_predictor(num_anchors):
    return nn.Conv2D(num_anchors * 4, kernel_size=3, padding=1)
    
    
   
#不同层预测出来的类别和边框的尺寸是不一样的，下面定义了一种合并方式
#每一层输出的是shape是(batch,C,H,W),不同层的输出，只有batch是一致的，其他三个值都不一样
#下面的函数把(batch,C,H,W)转换成(batch,HxWxC)
#最后在dim=1上连接
#注意mx.nd.flatten()的功能和numpy.flatten()不同，mx.nd.flatten()会保留维度0，只合并后面的维度
def flatten_pred(pred):
    return pred.transpose((0, 2, 3, 1)).flatten()

def concat_preds(preds):
    return nd.concat(*[flatten_pred(p) for p in preds], dim=1)    

    
def blk_forward(X, blk, size, ratio, cls_predictor, bbox_predictor):
    Y = blk(X) #提取特征
    #anchors = contrib.ndarray.MultiBoxPrior(Y, sizes=size, ratios=ratio,steps=(1,1),offsets=(0.5,0.5)) #获得anchor
    anchors = contrib.ndarray.MultiBoxPrior(Y, sizes=size, ratios=ratio) #获得anchor
    cls_preds = cls_predictor(Y) #预测类别 （这不是上面定义的函数，而是其具体实现，即一个卷积层）
    bbox_preds = bbox_predictor(Y) #预测边界框 （这不是上面定义的函数，而是其具体实现，即一个卷积层）
    return (Y, anchors, cls_preds, bbox_preds)




class BACKBONE(nn.Block):
    def __init__(self):
        super(BACKBONE, self).__init__()
        self.stageI = nn.Sequential()
        self.stageI.add(
            conv_layer(64,3,1,1),
            nn.MaxPool2D(pool_size=3,strides=2,padding=1),
            conv_layer(96,3,1,1),
            nn.MaxPool2D(pool_size=3,strides=2,padding=1),
            conv_layer(128,3,1,1),
            nn.MaxPool2D(pool_size=3,strides=2,padding=1),
        )
        return
    def forward(self, X):
        y1 = self.stageI(X)
        return y1
    
def calc_anchor_sizes(ranges, num):
    r0,r1 = ranges
    step_size = (r1 - r0) / float(num)
    sizes = [ (r0 + k * step_size, r0 + (k + 1) * step_size) for k in range(num)]
    #print(sizes)
    return sizes

class SSD(nn.Block):
    def __init__(self, num_classes, anchor_sizes = None, anchor_ratios = None, backbone=None, **kwargs):
        super(SSD, self).__init__(**kwargs)
        self.num_classes = num_classes
        
        if anchor_sizes is None:
            self.anchor_sizes = calc_anchor_sizes((0.1,0.96),6)
        else:
            self.anchor_sizes = anchor_sizes
        
        if anchor_ratios is None:
            self.anchor_ratios = ((1, 2, 0.5),(1, 2, 0.5),(1, 2, 0.5),(1, 2, 0.5),(1, 2, 0.5),(1,2,0.5))
        else:
            self.anchor_ratios = anchor_ratios
            
        self.num_anchors = len(self.anchor_sizes[0]) + len(self.anchor_ratios[0]) - 1

        self.stage_0, self.stage_1, self.stage_2, self.stage_3, self.stage_4, self.stage_5 = nn.Sequential(),nn.Sequential(),nn.Sequential(),nn.Sequential(),nn.Sequential(), nn.Sequential()


        backbone = BACKBONE()

        self.stage_0.add( backbone, cls_predictor(self.num_anchors, self.num_classes), bbox_predictor(self.num_anchors))
        self.stage_1.add(
            #INCEPTION_BLOCK(256)
            conv_layer(128,3,1,1),
            conv_layer(128,3,1,1),
        )
        
        self.stage_2.add( down_sample_blk(128*2), cls_predictor(self.num_anchors, self.num_classes), bbox_predictor(self.num_anchors) )
        self.stage_3.add( down_sample_blk(128*2), cls_predictor(self.num_anchors, self.num_classes), bbox_predictor(self.num_anchors) )
        self.stage_4.add( down_sample_blk(128*2), cls_predictor(self.num_anchors, self.num_classes), bbox_predictor(self.num_anchors) )
        self.stage_5.add( down_sample_blk(128*2), cls_predictor(self.num_anchors, self.num_classes), bbox_predictor(self.num_anchors) )

        self.stage_0.initialize(init=mx.initializer.Xavier())
        self.stage_1.initialize(init=mx.initializer.Xavier())
        self.stage_2.initialize(init=mx.initializer.Xavier())
        self.stage_3.initialize(init=mx.initializer.Xavier())
        self.stage_4.initialize(init=mx.initializer.Xavier())
        self.stage_5.initialize(init=mx.initializer.Xavier())
        
        
        return
    def forward(self, X):
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        #print(X.shape)
        X,anchors[0], cls_preds[0], bbox_preds[0] = blk_forward(X, self.stage_0[0], self.anchor_sizes[0], self.anchor_ratios[0], self.stage_0[1], self.stage_0[2])
        #print(X.shape)
        X = self.stage_1(X)
        #print(X.shape)
        X,anchors[1], cls_preds[1], bbox_preds[1] = blk_forward(X, self.stage_2[0], self.anchor_sizes[1], self.anchor_ratios[1], self.stage_2[1], self.stage_2[2])
        #print(X.shape)
        X,anchors[2], cls_preds[2], bbox_preds[2] = blk_forward(X, self.stage_3[0], self.anchor_sizes[2], self.anchor_ratios[2], self.stage_3[1], self.stage_3[2])
        #print(X.shape)
        X,anchors[3], cls_preds[3], bbox_preds[3] = blk_forward(X, self.stage_4[0], self.anchor_sizes[3], self.anchor_ratios[3], self.stage_4[1], self.stage_4[2])
        #print(X.shape)
        X,anchors[4], cls_preds[4], bbox_preds[4] = blk_forward(X, self.stage_5[0], self.anchor_sizes[4], self.anchor_ratios[4], self.stage_5[1], self.stage_5[2])
        # reshape函数中的0表示保持批量大小不变
        #print(X.shape)
        return (nd.concat(*anchors, dim=1), concat_preds(cls_preds).reshape( (0, -1, self.num_classes + 1)), concat_preds(bbox_preds))

if 0:
    net = SSD(10)
    ctx = mx.gpu()
    net.collect_params().reset_ctx(ctx)
    x = nd.zeros((32,3,300,300),ctx=ctx)
    y = net(x)
    print(y[0].shape)
    print(y[1].shape)
    print(y[2].shape)
    
