#coding=utf-8
from mxnet import contrib, nd,gluon,autograd
from mxnet.gluon import  nn
import mxnet as mx
from mxnet.gluon.model_zoo import vision
from gluoncv.model_zoo import vgg16_atrous_300




class NormScale(nn.HybridBlock):
    def __init__(self, channel, scale, prefix='scale_'):
        super(NormScale, self).__init__(prefix=prefix)
        with self.name_scope():
            self._channel = channel
            self._scale = scale
            self.weight = self.params.get('scale', shape=(1, channel, 1, 1),
                                          init=mx.init.Constant(scale), wd_mult=0.1)

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = F.L2Normalization(x, mode="channel")
        x = F.broadcast_mul(lhs=x, rhs=self.weight.data())
        return x


def ssd_extra_one(channels, stride, padding):
    layer = nn.Sequential()
    layer.add(
        nn.Conv2D(channels//2, 1, strides=1, padding=0, dilation=1,weight_initializer=mx.init.Xavier()),
        nn.BatchNorm(),
        nn.Activation("relu"),
        nn.Conv2D(channels , 3, strides=stride, padding=padding, dilation=1,weight_initializer=mx.init.Xavier()),
        nn.BatchNorm(),
        nn.Activation("relu") )
    return layer



    
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
def flatten_pred(F,pred):
    return F.flatten( F.transpose(pred,(0, 2, 3, 1)) )

def concat_preds(F,preds):
    return F.concat(*[flatten_pred(F,p) for p in preds], dim=1)

class BackboneResnet34(nn.Block):
    def __init__(self):
        super(BackboneResnet34, self).__init__()
        self.info = "resnet34"
        self.features = vision.resnet34_v1(pretrained=True).features[0:-2]
        self.append = nn.Sequential()
        self.append.add(
            nn.Conv2D(1024,kernel_size=3, padding=6,dilation=6,weight_initializer=mx.init.Xavier()),
            nn.BatchNorm(),
            nn.Activation("relu"),
            nn.Conv2D(1024, kernel_size=1, padding=0, dilation=1, weight_initializer=mx.init.Xavier()),
            nn.BatchNorm(),
            nn.Activation("relu")
        )
        self.extra = nn.Sequential()
        self.extra.add(
            ssd_extra_one(channels=512, stride=2, padding=1),
            ssd_extra_one(channels=256, stride=2, padding=1),
            ssd_extra_one(channels=256, stride=1, padding=0), #smaller feature map
            ssd_extra_one(channels=256, stride=1, padding=0),
        )
        self.features.collect_params().setattr("lr_mult",0.1)
        self.extra.initialize()
        self.append.initialize()
        return
    def forward(self, X):
        outputs = []
        for layer in self.features[0:-1]:
            X = layer(X)
        outputs.append(X)
        X = self.features[-1](X)
        X = self.append(X)
        outputs.append(X)

        for layer in self.extra:
            X = layer(X)
            outputs.append(X)
        return outputs


def calc_anchor_sizes(ranges, num):
    r0,r1 = ranges
    step_size = (r1 - r0) / float(num)
    sizes = [ (r0 + k * step_size, r0 + (k + 1) * step_size) for k in range(num)]
    #print(sizes)
    return sizes


from gluoncv.model_zoo.ssd.target import SSDTargetGenerator
def ssd_forward_one(Y, size, ratio, cls_predictor, bbox_predictor):
   # Y = blk(X) #提取特征
    #anchors = contrib.ndarray.MultiBoxPrior(Y, sizes=size, ratios=ratio,steps=(1,1),offsets=(0.5,0.5)) #获得anchor
    #anchors = contrib.ndarray.MultiBoxPrior(Y, sizes=size, ratios=ratio) #获得anchor  (x0,y0,x1,y1)
    anchors = contrib.sym.MultiBoxPrior(Y, sizes=size, ratios=ratio) #获得anchor  (x0,y0,x1,y1)
    #for row in range(anchors[0].shape[0]):
    #    x0,y0,x1,y1 = anchors[0,row].asnumpy().tolist()
    #    w,h = x1 - x0, y1 - y0
    #    print('xx size={} ratio = {}'.format(w*h, w/h))
    cls_preds = cls_predictor(Y) #预测类别 （这不是上面定义的函数，而是其具体实现，即一个卷积层）
    bbox_preds = bbox_predictor(Y) #预测边界框 （这不是上面定义的函数，而是其具体实现，即一个卷积层）
    return (anchors, cls_preds, bbox_preds)

def get_vgg16_300():
    pretrained = vgg16_atrous_300(pretrained=True)
    body = gluon.nn.HybridSequential()
    with body.name_scope():
        body.add(pretrained)
    return body

class SSD_CUSTOM(nn.Block):
    def __init__(self, num_classes, anchor_sizes = None, anchor_ratios = None , **kwargs):
        super(SSD_CUSTOM, self).__init__(**kwargs)
        self.num_classes = num_classes
        
        if anchor_sizes is None:
            self.anchor_sizes = calc_anchor_sizes((0.2,0.96),6)
        else:
            self.anchor_sizes = anchor_sizes
        
        if anchor_ratios is None:
            self.anchor_ratios = ( (1, 2, 0.5), (1, 2, 3, 0.5, 1.0 / 3), (1, 2, 3, 0.5, 1.0 / 3), \
                (1, 2, 3, 0.5, 1.0 / 3), (1, 2, 0.5),(1, 2, 0.5))
        else:
            self.anchor_ratios = anchor_ratios
            
        num_anchors = []
        for size, ratio in zip(self.anchor_sizes, self.anchor_ratios):
            num_anchors.append( len(size) + len(ratio) - 1 )

        self.stage_0, self.stage_1, self.stage_2, self.stage_3, self.stage_4, self.stage_5 = nn.Sequential(),nn.Sequential(),nn.Sequential(),nn.Sequential(),nn.Sequential(), nn.Sequential()


        backbone = BackboneResnet34()
        print("using pretrained backbone:",backbone.info)
                

        self.backbone = backbone
        with self.stage_0.name_scope():
            self.stage_0.add(
                             cls_predictor(num_anchors[0], self.num_classes), bbox_predictor(num_anchors[0]))
        self.stage_1.add(
                         cls_predictor(num_anchors[1], self.num_classes), bbox_predictor(num_anchors[1]))
        self.stage_2.add(
                         cls_predictor(num_anchors[2], self.num_classes), bbox_predictor(num_anchors[2]))
        self.stage_3.add(
                         cls_predictor(num_anchors[3], self.num_classes), bbox_predictor(num_anchors[3]))
        self.stage_4.add(
                         cls_predictor(num_anchors[4], self.num_classes), bbox_predictor(num_anchors[4]))
        self.stage_5.add(
                         cls_predictor(num_anchors[5], self.num_classes), bbox_predictor(num_anchors[5]))

        self.stage_0.initialize(init=mx.initializer.Xavier())
        self.stage_1.initialize(init=mx.initializer.Xavier())
        self.stage_2.initialize(init=mx.initializer.Xavier())
        self.stage_3.initialize(init=mx.initializer.Xavier())
        self.stage_4.initialize(init=mx.initializer.Xavier())
        self.stage_5.initialize(init=mx.initializer.Xavier())
        
        
        return
    def forward(self, X):
        anchors, cls_preds, bbox_preds = [None] * 6, [None] * 6, [None] * 6
        # print(X.shape)
        Xs = self.backbone(X)
        anchors[0], cls_preds[0], bbox_preds[0] = ssd_forward_one(Xs[0], self.anchor_sizes[0],
                                                                  self.anchor_ratios[0], self.stage_0[0],
                                                                  self.stage_0[1])
        # print(anchors[0].shape, cls_preds[0].shape, bbox_preds[0].shape)
        anchors[1], cls_preds[1], bbox_preds[1] = ssd_forward_one(Xs[1], self.anchor_sizes[1],
                                                                  self.anchor_ratios[1], self.stage_1[0],
                                                                  self.stage_1[1])
        # print(X.shape)
        anchors[2], cls_preds[2], bbox_preds[2] = ssd_forward_one(Xs[2], self.anchor_sizes[2],
                                                                  self.anchor_ratios[2], self.stage_2[0],
                                                                  self.stage_2[1])
        # print(X.shape)
        anchors[3], cls_preds[3], bbox_preds[3] = ssd_forward_one(Xs[3], self.anchor_sizes[3],
                                                                  self.anchor_ratios[3], self.stage_3[0],
                                                                  self.stage_3[1])
        # print(X.shape)
        anchors[4], cls_preds[4], bbox_preds[4] = ssd_forward_one(Xs[4], self.anchor_sizes[4],
                                                                  self.anchor_ratios[4], self.stage_4[0],
                                                                  self.stage_4[1])
        # print(X.shape)
        anchors[5], cls_preds[5], bbox_preds[5] = ssd_forward_one(Xs[5], self.anchor_sizes[5],
                                                                  self.anchor_ratios[5], self.stage_5[0],  # last bug!!
                                                                  self.stage_5[1])
        # reshape函数中的0表示保持批量大小不变
        # print(X.shape)
        return (nd.concat(*anchors, dim=1), concat_preds(cls_preds).reshape((0, -1, self.num_classes + 1)),
                concat_preds(bbox_preds))
#############################################################################
#############################################################################
import gluoncv as gcv
class Postpreocess(gluon.HybridBlock):
    def __init__(self,num_classes,nms_thresh=0.45, nms_topk=400, post_nms=100):
        super(Postpreocess,self).__init__()
        self.num_classes = num_classes
        self.nms_thresh = nms_thresh
        self.nms_topk = nms_topk
        self.post_nms = post_nms
        self.bbox_decoder = gcv.nn.coder.NormalizedBoxCenterDecoder(stds=[0.1,0.1,0.2,0.2])
        self.cls_decoder = gcv.nn.coder.MultiPerClassDecoder(self.num_classes + 1, thresh=0.01)
        self.corner2center = gcv.nn.bbox.BBoxCornerToCenter(split=False)
        return


    def hybrid_forward(self, F, anchors, pred_classes, pred_bboxes):
        """
        :param anchors: (1, num-of-anchor, 4), anchors[0,0,:] = x0,y0,x1,y1
        :param pred_classes: (batch-size, num-of-anchor, num-of-classes), including background
        :param pred_bboxes: (batch-size, num-of-anchor * 4)

        :param ids: (batch-size, num-of-found, 1)  class id for each found
        :param scores: (batch-size, num-of-found, 1)  class score for each found
        :param bboxes: (batch-size, num-of-found, 4)  coordinates of each found  (x0,y0,x1,y1) with norm w/h
        """
        anchors = self.corner2center(anchors)
        pred_bboxes = F.reshape(pred_bboxes, (0,-1,4))
        bboxes = self.bbox_decoder(pred_bboxes, anchors)

        cls_ids, scores = self.cls_decoder(F.softmax(pred_classes,axis=-1))


        results = []
        for i in range(self.num_classes):
            cls_id = cls_ids.slice_axis(axis=-1, begin=i, end=i + 1)
            score = scores.slice_axis(axis=-1, begin=i, end=i + 1)
            # per class results
            per_result = F.concat(*[cls_id, score, bboxes], dim=-1)
            results.append(per_result)
        result = F.concat(*results, dim=1)
        if self.nms_thresh > 0 and self.nms_thresh < 1:
            result = F.contrib.box_nms(
                result, overlap_thresh=self.nms_thresh, topk=self.nms_topk, valid_thresh=0.01,
                id_index=0, score_index=1, coord_start=2, force_suppress=False)
            if self.post_nms > 0:
                result = result.slice_axis(axis=1, begin=0, end=self.post_nms)

        ids = F.slice_axis(result, axis=2, begin=0, end=1)
        scores = F.slice_axis(result, axis=2, begin=1, end=2)
        bboxes = F.slice_axis(result, axis=2, begin=2, end=6)
        return ids, scores, bboxes

class SSD(nn.HybridBlock):
    def __init__(self, num_classes, anchor_sizes=None, anchor_ratios=None, backbone=None, **kwargs):
        super(SSD, self).__init__(**kwargs)
        self.num_classes = num_classes

        if anchor_sizes is None:
            self.anchor_sizes = calc_anchor_sizes((0.2, 0.9), 6)
        else:
            self.anchor_sizes = anchor_sizes

        if anchor_ratios is None:
            self.anchor_ratios = (
                    (1, 2, 0.5), (1, 2, 3, 0.5, 1.0 / 3), (1, 2, 3, 0.5, 1.0 / 3), (1, 2, 3, 0.5, 1.0 / 3), (1, 2, 0.5), (1, 2, 0.5))
        else:
            self.anchor_ratios = anchor_ratios

        num_anchors = []
        for size, ratio in zip(self.anchor_sizes, self.anchor_ratios):
            num_anchors.append( len(size) + len(ratio) - 1 )

        self.stage_0, self.stage_1, self.stage_2, self.stage_3, self.stage_4, self.stage_5 = nn.HybridSequential(prefix="decode_0"), nn.HybridSequential(), nn.HybridSequential(), nn.HybridSequential(), nn.HybridSequential(), nn.HybridSequential()


        backbone = get_vgg16_300()
        print("using pretrained backbone: vgg16_atrous_300")
        backbone.collect_params().setattr("lr_mult", 0.4)

        self.backbone = backbone
        with self.stage_0.name_scope():
            self.stage_0.add(
                             cls_predictor(num_anchors[0], self.num_classes), bbox_predictor(num_anchors[0]))
        self.stage_1.add(
                         cls_predictor(num_anchors[1], self.num_classes), bbox_predictor(num_anchors[1]))
        self.stage_2.add(
                         cls_predictor(num_anchors[2], self.num_classes), bbox_predictor(num_anchors[2]))
        self.stage_3.add(
                         cls_predictor(num_anchors[3], self.num_classes), bbox_predictor(num_anchors[3]))
        self.stage_4.add(
                         cls_predictor(num_anchors[4], self.num_classes), bbox_predictor(num_anchors[4]))
        self.stage_5.add(
                         cls_predictor(num_anchors[5], self.num_classes), bbox_predictor(num_anchors[5]))

        # self.stage_0[0].initialize(init=mx.initializer.Xavier())
        self.stage_0.initialize(init=mx.initializer.Xavier())
        self.stage_1.initialize(init=mx.initializer.Xavier())
        self.stage_2.initialize(init=mx.initializer.Xavier())
        self.stage_3.initialize(init=mx.initializer.Xavier())
        self.stage_4.initialize(init=mx.initializer.Xavier())
        self.stage_5.initialize(init=mx.initializer.Xavier())

        self.postprocess = Postpreocess(self.num_classes)

        return

    def hybrid_forward(self, F, X):
        anchors, cls_preds, bbox_pred = [None] * 6, [None] * 6, [None] * 6
        # print(X.shape)
        Xs = self.backbone(X)
        anchors[0], cls_preds[0], bbox_pred[0] = ssd_forward_one(Xs[0], self.anchor_sizes[0],
                                                                 self.anchor_ratios[0], self.stage_0[0],
                                                                 self.stage_0[1])
       # print(anchors[0].shape, cls_preds[0].shape, bbox_preds[0].shape)
        anchors[1], cls_preds[1], bbox_pred[1] = ssd_forward_one(Xs[1], self.anchor_sizes[1],
                                                                 self.anchor_ratios[1], self.stage_1[0],
                                                                 self.stage_1[1])
        # print(X.shape)
        anchors[2], cls_preds[2], bbox_pred[2] = ssd_forward_one(Xs[2], self.anchor_sizes[2],
                                                                 self.anchor_ratios[2], self.stage_2[0],
                                                                 self.stage_2[1])
        # print(X.shape)
        anchors[3], cls_preds[3], bbox_pred[3] = ssd_forward_one(Xs[3],  self.anchor_sizes[3],
                                                                 self.anchor_ratios[3], self.stage_3[0],
                                                                 self.stage_3[1])
        # print(X.shape)
        anchors[4], cls_preds[4], bbox_pred[4] = ssd_forward_one(Xs[4], self.anchor_sizes[4],
                                                                 self.anchor_ratios[4], self.stage_4[0],
                                                                 self.stage_4[1])
        # print(X.shape)
        anchors[5], cls_preds[5], bbox_pred[5] = ssd_forward_one(Xs[5],  self.anchor_sizes[5],
                                                                 self.anchor_ratios[5], self.stage_5[0],  #last bug!!
                                                                 self.stage_5[1])
        # reshape函数中的0表示保持批量大小不变
        # print(X.shape)
        anchors = F.concat(*anchors, dim=1)
        cls_preds = F.reshape( concat_preds(F,cls_preds), (0, -1, self.num_classes + 1))
        bbox_pred = concat_preds(F,bbox_pred)
        if autograd.is_training(): #for hybridBlock, must call hybridize() to make this take effect!!
            return anchors, cls_preds, bbox_pred
        return self.postprocess( anchors, cls_preds, bbox_pred )
        


if 0:
    net = SSD_CUSTOM(100)
    ctx = mx.gpu()
    net.collect_params().reset_ctx(ctx)
    x = nd.zeros((32,3,256*2,256*2),ctx=ctx)
    y = net(x)


if 0:
    #net = get_resnet_34()
    net = vision.resnet34_v1(pretrained=True).features
    ctx = mx.gpu()
    net.collect_params().reset_ctx(ctx)
    x = nd.zeros((32,3,256*2,256*2),ctx=ctx)
    y = net(x)
    for layer in net:
        x = layer(x)
        print(layer.name, x.shape)
    data = mx.sym.var("data")
    for sym in net(data).get_internals():
        print(sym)


if 0:
   # net = model_zoo.get_model('ssd_300_vgg16_atrous_voc', pretrained=False, pretrained_base=False)
   #net = vgg16_atrous_300(pretrained=True)
    net = BackboneResnet34()
    ctx = mx.gpu()
    #net.initialize()
    net.collect_params().reset_ctx(ctx)
    x = nd.zeros((32,3,300,300),ctx=ctx)
    y = net(x)
    #print(net,' net')
    for k in range(len(y)):
        print(k,y[k].shape)



if 0:
    net = SSD_CUSTOM(21)
    ctx = mx.gpu()
    net.collect_params().reset_ctx(ctx)
    x = nd.zeros((32,3,300,300),ctx=ctx)
    y = net(x)
    print(net.collect_params())
    print(y[0].shape)
    print(y[1].shape)
    print(y[2].shape)
    
