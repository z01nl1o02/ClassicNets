from mxnet.gluon import nn
import mxnet as mx
from mxnet import nd

def conv_layer(out_channel,kernel_size, padding,strides):
    stages = nn.Sequential()
    stages.add(nn.Conv2D(out_channel, kernel_size=kernel_size, padding=padding, strides = strides),
    #nn.BatchNorm(), nn.Activation("relu")
    )
    return stages

class FIRE(nn.Block):
    def __init__(self,sizes):
        super(FIRE,self).__init__()
        s1,e1,e2 = sizes
        self.squeeze_1x1 = conv_layer(s1, 1, 0, 1)
        self.expand_1x1 =  conv_layer(e1, 1, 0, 1)
        self.expand_3x3 =  conv_layer(e1, 3, 1, 1)
        return
    def forward(self,X):
        y1 = self.squeeze_1x1(X)
        y2 = self.expand_1x1(y1)
        y3 = self.expand_3x3(y1)
        Y = nd.concat(y2,y3,dim=1) 
        return Y


class SQUEEZENET(nn.Block):
    def __init__(self,number_classes, config_first_conv = (96,7,0,2) ):
        super(SQUEEZENET,self).__init__()
        self.stages = nn.Sequential()
        self.stages.add( conv_layer(*config_first_conv),
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
                conv_layer(number_classes, 1, 0, 1),
                nn.GlobalAvgPool2D(),
                nn.Flatten())
        return

    def forward(self, X):
        return self.stages(X)

def load(num_classes,config_first_conv = (96,7,0,2)):
    net = SQUEEZENET(num_classes,config_first_conv )
    net.initialize(mx.initializer.Xavier())
    return net
    
if 0:
    #import os
    #os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = 0
    from mxnet import nd
    ctx = mx.gpu(0)
    X = nd.random.uniform(0,1,(1,3,32,32),ctx=ctx)
    net = load(21,(96,3,1,1))
    net.collect_params().reset_ctx(ctx=ctx)
    #print(net)
    Y = net(X)[:,:,0,0]
    print(X.shape, Y.shape)
