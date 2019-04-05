import mxnet as mx
from mxnet import gluon,nd
from mxnet.gluon import nn

class DOWNBLOCK(nn.Block):
    def __init__(self,num_filters):
        super(DOWNBLOCK,self).__init__()
        self.stageA = nn.Sequential()
        for num in num_filters:
            self.stageA.add(
                nn.BatchNorm(),
                nn.Conv2D(num,kernel_size=3,strides=1,padding=1,activation="relu"),
            )
        self.stageB = nn.Sequential()
        self.stageB.add(
            nn.MaxPool2D(pool_size=2,strides=2)
        )
    def forward(self, X):
        outA = self.stageA(X)
        outB = self.stageB(outA)
        return outA,outB

class UPBLOCK(nn.Block):
    def __init__(self, num_filters):
        super(UPBLOCK,self).__init__()
        self.stageA = nn.Sequential()
        for num in num_filters:
            self.stageA.add(
                nn.BatchNorm(),
                nn.Conv2D(num,kernel_size=3,strides=1,padding=1,activation="relu"),
            )
        self.stageB = nn.Sequential()
        self.stageB.add(
            nn.Conv2D(num_filters[-1],kernel_size=1,strides=1,padding=0,activation="relu"),
        )
        return
    def forward(self, X):
        Y = self.stageA(X)
        _,_,H,W = Y.shape
        Y = nd.contrib.BilinearResize2D(Y,height=H*2, width=W*2)
        Y = self.stageB(Y)
        return Y


class UNET(nn.Block):
    def __init__(self,num_classes):
        super(UNET,self).__init__()

        self.downA = DOWNBLOCK((64,64))
        self.downB = DOWNBLOCK((128,128,128))
        self.downC = DOWNBLOCK((256,256,256))
        self.downD = DOWNBLOCK((512,512,512))
        self.downE = DOWNBLOCK((1024,1024,1024))

        self.upA = UPBLOCK((512,512,512))
        self.upB = UPBLOCK((256,256,256))
        self.upC = UPBLOCK((128,128,128))
        self.upD = UPBLOCK((64,64,64,num_classes))
        return
    def forward(self,X):
        down_out_A0, down_out_A1 = self.downA(X)
        down_out_B0, down_out_B1 = self.downB(down_out_A1)
        down_out_C0, down_out_C1 = self.downC(down_out_B1)
        down_out_D0, down_out_D1 = self.downD(down_out_C1)
        down_out_E0,_ = self.downE(down_out_D1)


        up_out_A = nd.concat(self.upA(down_out_E0),down_out_D0,dim=1)
        up_out_B = nd.concat(self.upB(up_out_A), down_out_C0,dim=1)
        up_out_C = nd.concat(self.upC(up_out_B), down_out_B0,dim=1)
        up_out_D = self.upD(up_out_C)
        return up_out_D

def get_net(num_classes):
    net = UNET(num_classes)
    net.initialize(mx.init.Xavier())
    return net

if 0:
    #import os
    #os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = 0
    ctx = mx.gpu(0)
    X = nd.random.uniform(0,1,(1,3,512,512),ctx=ctx)
    net = get_net(21)
    net.initialize(ctx=ctx)
    print(net)
    Y = net(X)
    print(X.shape, Y.shape)





