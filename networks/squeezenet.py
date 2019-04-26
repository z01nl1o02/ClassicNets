from mxnet.gluon import nn
import mxnet as mx
from mxnet import nd

class FIRE(nn.Block):
    def __init__(self,sizes):
        super(FIRE,self).__init__()
        s1,e1,e2 = sizes
        self.squeeze_1x1 = nn.Conv2D(s1,kernel_size=1,padding=0,strides=1,activation="relu")
        self.expand_1x1 = nn.Conv2D(e1,kernel_size=1,padding=0,strides=1,activation='relu')
        self.expand_3x3 = nn.Conv2D(e2,kernel_size=3,padding=1,strides=1,activation="relu")
        return
    def forward(self,X):
        y1 = self.squeeze_1x1(X)
        y2 = self.expand_1x1(y1)
        y3 = self.expand_3x3(y2)
        Y = nd.concat(y2,y3,dim=1) 
        return Y


class SQUEEZENET(nn.Block):
    def __init__(self,number_classes):
        super(SQUEEZENET,self).__init__()
        self.stages = nn.Sequential()
        self.stages.add( nn.Conv2D(96,kernel_size=7,padding=0,strides=2,activation="relu"),
                         nn.MaxPool2D(pool_size=3,strides=2))
        self.stages.add(
                FIRE( (16,64,64) ),
                FIRE( (16,64,64) ),
                FIRE( (32,128,128) ),
                nn.MaxPool2D(pool_size=3,strides=2),
                FIRE( (32,128,128) ),
                FIRE( (48,192,192) ),
                FIRE( (48,192,192) ),
                FIRE( (64,256,256) ),
                nn.MaxPool2D(pool_size=3,strides=2),
                FIRE( (64,256,256) ),
                nn.Dropout(0.5),
                nn.Conv2D(number_classes,kernel_size=1,padding=0,strides=1,activation="relu"),
                nn.GlobalAvgPool2D(),
        )
        return
    def forward(self,X):            
        return self.stages(X)

def load(num_classes):
    net = SQUEEZENET(num_classes)
    net.initialize(mx.initializer.Xavier())
    return net
    
if 0:
    #import os
    #os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = 0
    from mxnet import nd
    ctx = mx.gpu(0)
    X = nd.random.uniform(0,1,(1,3,35,35),ctx=ctx)
    net = load(21)
    net.collect_params().reset_ctx(ctx=ctx)
    #print(net)
    Y = net(X)[:,:,0,0]
    print(X.shape, Y.shape)
