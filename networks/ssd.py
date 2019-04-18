from mxnet import contrib, image, nd,gluon,autograd, init
from mxnet.gluon import loss as gloss, nn
import mxnet as mx
import numpy as np



def down_sample_blk(num_channels):
    blk = nn.Sequential()
    for _ in range(2):
        blk.add(nn.Conv2D(num_channels, kernel_size=3, padding=1),
                nn.BatchNorm(in_channels=num_channels),
                nn.Activation('relu'))
    blk.add(nn.MaxPool2D(2))
    return blk
    
    
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
    anchors = contrib.ndarray.MultiBoxPrior(Y, sizes=size, ratios=ratio) #获得anchor
    cls_preds = cls_predictor(Y) #预测类别 （这不是上面定义的函数，而是其具体实现，即一个卷积层）
    bbox_preds = bbox_predictor(Y) #预测边界框 （这不是上面定义的函数，而是其具体实现，即一个卷积层）
    return (Y, anchors, cls_preds, bbox_preds)

    
class SSD(nn.Block):
    def __init__(self, num_classes, anchor_sizes = None, anchor_ratios = None, backbone="vgg-11", **kwargs):
        super(SSD, self).__init__(**kwargs)
        self.num_classes = num_classes
        
        if anchor_sizes is None:
            #self.anchor_sizes = ((0.2, 0.272), (0.37, 0.447), (0.54, 0.619), (0.71, 0.79),(0.88, 0.961))
            self.anchor_sizes = [[0.1, 0.16], [0.16, 0.22], [0.22, 0.28], [0.28, 0.34], [0.34, 0.4]]
        else:
            self.anchor_sizes = anchor_sizes
        
        if anchor_ratios is None:
            self.anchor_ratios = ((1, 2, 0.5),(1, 2, 0.5),(1, 2, 0.5),(1, 2, 0.5),(1, 2, 0.5))
        else:
            self.anchor_ratios = anchor_ratios
            
        self.num_anchors = len(self.anchor_sizes[0]) + len(self.anchor_ratios[0]) - 1

               
        self.stage_0, self.stage_1, self.stage_2, self.stage_3, self.stage_4 = nn.Sequential(),nn.Sequential(),nn.Sequential(),nn.Sequential(),nn.Sequential()
 

        #self.stage_0 = nn.Sequential() 
        backbone = nn.Sequential() 
        pretrained = gluon.model_zoo.vision.vgg11(pretrained=True)
        for layer in pretrained.features[0:-10]:
            backbone.add(layer)
        self.stage_0.add( backbone, cls_predictor(self.num_anchors, self.num_classes), bbox_predictor(self.num_anchors))
        
        
        self.stage_1.add( down_sample_blk(512), cls_predictor(self.num_anchors, self.num_classes), bbox_predictor(self.num_anchors) )
        self.stage_2.add( down_sample_blk(512), cls_predictor(self.num_anchors, self.num_classes), bbox_predictor(self.num_anchors) )
        self.stage_3.add( down_sample_blk(512), cls_predictor(self.num_anchors, self.num_classes), bbox_predictor(self.num_anchors) )
        self.stage_4.add( down_sample_blk(512), cls_predictor(self.num_anchors, self.num_classes), bbox_predictor(self.num_anchors) )
        
        
        self.stage_0[-2:].initialize(init=mx.initializer.Xavier())
        self.stage_1.initialize(init=mx.initializer.Xavier())
        self.stage_2.initialize(init=mx.initializer.Xavier())
        self.stage_3.initialize(init=mx.initializer.Xavier())
        self.stage_4.initialize(init=mx.initializer.Xavier())
        
        
        return
    def forward(self, X):
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        #print(X.shape)
        X,anchors[0], cls_preds[0], bbox_preds[0] = blk_forward(X, self.stage_0[0], self.anchor_sizes[0], self.anchor_ratios[0], self.stage_0[1], self.stage_0[2])
        #print(X.shape)
        X,anchors[1], cls_preds[1], bbox_preds[1] = blk_forward(X, self.stage_1[0], self.anchor_sizes[1], self.anchor_ratios[1], self.stage_1[1], self.stage_1[2])
        #print(X.shape)
        X,anchors[2], cls_preds[2], bbox_preds[2] = blk_forward(X, self.stage_2[0], self.anchor_sizes[2], self.anchor_ratios[2], self.stage_2[1], self.stage_2[2])
        #print(X.shape)
        X,anchors[3], cls_preds[3], bbox_preds[3] = blk_forward(X, self.stage_3[0], self.anchor_sizes[3], self.anchor_ratios[3], self.stage_3[1], self.stage_3[2])
        #print(X.shape)
        X,anchors[4], cls_preds[4], bbox_preds[4] = blk_forward(X, self.stage_4[0], self.anchor_sizes[4], self.anchor_ratios[4], self.stage_4[1], self.stage_4[2])
        # reshape函数中的0表示保持批量大小不变
        #print(X.shape)
        return (nd.concat(*anchors, dim=1), concat_preds(cls_preds).reshape( (0, -1, self.num_classes + 1)), concat_preds(bbox_preds))
 
if 0:
    net = SSD(10)
    ctx = mx.gpu()
    net.collect_params().reset_ctx(ctx)
    x = nd.zeros((32,3,256,256),ctx=ctx)
    y = net(x)
    print(y[0].shape)
    print(y[1].shape)
    print(y[2].shape)
    