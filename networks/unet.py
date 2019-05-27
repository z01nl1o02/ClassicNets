import mxnet as mx
from mxnet import gluon,nd
from mxnet.gluon import nn
from layers import SpatialDropout2D

class Upsampling(nn.Block):
    def __init__(self,channels, scale=2):
        super(Upsampling,self).__init__()
        with self.name_scope():
            self.scale = scale
            self.conv = nn.Conv2D(channels,kernel_size=3,strides=1,padding=1)
        return
    def forward(self,X):
        _,_,H,W = X.shape
        y = nd.contrib.BilinearResize2D(X, height=H * self.scale, width=W * self.scale)
        #print(y.shape, X.shape)
        return self.conv(y)

class ConvBNRelu(nn.Block):
    def __init__(self,filters):
        super(ConvBNRelu,self).__init__()
        with self.name_scope():
            self.layers = nn.Sequential()
            for f in filters:
                self.layers.add(
                    nn.Conv2D(f,kernel_size=3,strides=1,padding=1),
                    nn.BatchNorm(),
                    nn.Activation("relu"),
                )
        return
    def forward(self, X):
        return self.layers(X)


class UNET(nn.Block):
    def __init__(self,num_classes):
        super(UNET,self).__init__()
        with self.name_scope():
            self.d1 = ConvBNRelu((8,8))

            self.d2d = nn.MaxPool2D(3,2,1)
            self.d2 = ConvBNRelu((16,16))

            self.d3d = nn.MaxPool2D(3,2,1)
            self.d3 = ConvBNRelu((32,32))

            self.d4d = nn.MaxPool2D(3,2,1)
            self.d4 = ConvBNRelu((64,64))

            self.d5d = nn.MaxPool2D(3,2,1)
            self.dp = SpatialDropout2D()
            self.u5 = ConvBNRelu((64,64))


            self.u4u = Upsampling(64)
            self.u4 = ConvBNRelu((32,32))


            self.u3u = Upsampling(32)
            self.u3 = ConvBNRelu((16,16))


            self.u2u = Upsampling(16)
            self.u2 = ConvBNRelu((8,8))

            self.u1u = Upsampling(8)
            self.u1 = ConvBNRelu((8,8))

            self.u0 = nn.Conv2D(num_classes,kernel_size=3,strides=1,padding=1)
        return
    def forward(self,X):
        d1 = self.d1(X)

        d2d = self.d2d(d1)
        d2 = self.d2(d2d)

        d3d = self.d3d(d2)
        d3 = self.d3(d3d)

        d4d = self.d4d(d3)
        d4 = self.d4(d4d)

        d5d = self.d5d(d4)
        u5 = self.u5(self.dp(d5d))

        u4 = self.u4( u5 )
        u4 = nd.concat(self.u4u(u4), d4)
        #print(u4.shape, d4.shape)

        u3 = self.u3(u4)
        u3 = nd.concat(self.u3u(u3),d3)
        #print(u3.shape, d3.shape)


        u2 = self.u2(u3)
        u2 = nd.concat(self.u2u(u2),d2)
        #print(u2.shape, d2.shape)


        u1 = self.u1(u2)
        u1 = nd.concat(self.u1u(u1),d1)
        #print(u1.shape, d1.shape)

        u0 = self.u0(u1)
        return u0



def get_net(num_classes):
    net = UNET(num_classes)
    net.initialize(mx.init.Xavier())
    return net

if 0:
    #import os
    #os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = 0
    ctx = mx.gpu(0)
    X = nd.random.uniform(0,1,(1,3,256,256),ctx=ctx)
    net = get_net(2)
    net.collect_params().reset_ctx(ctx)
    print(net)
    Y = net(X)
    print(X.shape, Y.shape)





