import mxnet as mx
from mxnet import gluon
import numpy as np
import cv2

#for tiny target batchnorm and dropout may cause much difference between train-stage and test-stage
#this fcn is based on vgg1 without batchnorm/dropout
class FCN(gluon.Block):
    def __init__(self, num_class,root='./'):
        super(FCN,self).__init__()
        backbone = gluon.model_zoo.vision.vgg11(pretrained=True,root = root)
        self.downlayer_2 = gluon.nn.Sequential()
        for layer in backbone.features[0:3]:
            self.downlayer_2.add(layer)
        
        self.downlayer_8 = gluon.nn.Sequential()
        for layer in backbone.features[3:11]:
            self.downlayer_8.add(layer)
            
        self.downlayer_32 = gluon.nn.Sequential()
        for layer in backbone.features[11:-4]:
           # print layer.name
            self.downlayer_32.add(layer)

        self.uplayer_8 = gluon.nn.Sequential()
        self.uplayer_8.add(
            gluon.nn.Conv2D(channels=256,kernel_size=3,strides=1,padding=1,activation="relu"),
            gluon.nn.Conv2D(channels=256,kernel_size=3,strides=1,padding=1,activation="relu")
        )
        self.uplayer_8.initialize(init=mx.initializer.Xavier())
        for layer in self.uplayer_8:
            layer.weight.lr_mult = 10.0


        self.uplayer_2 = gluon.nn.Sequential()
        self.uplayer_2.add(
            gluon.nn.Conv2D(channels=96,kernel_size=3,strides=1,padding=1,activation="relu"),
            gluon.nn.Conv2D(channels=64,kernel_size=3,strides=1,padding=1,activation="relu")
        )
        self.uplayer_2.initialize(init=mx.initializer.Xavier())
        for layer in self.uplayer_2:
            layer.weight.lr_mult = 10.0

        self.uplayer_1 = gluon.nn.Sequential()
        self.uplayer_1.add(
            gluon.nn.Conv2D(channels=32,kernel_size=3,strides=1,padding=1,activation="relu"),
            gluon.nn.Conv2D(channels=num_class,kernel_size=3,strides=1,padding=1,activation=None)
        )
        self.uplayer_1.initialize(init=mx.initializer.Xavier())
        for layer in self.uplayer_1:
            layer.weight.lr_mult = 10.0


        return

    def forward(self, x):
        out = x
        
        for layer in self.downlayer_2:
            out = layer(out)
        d2 = out
        
        for layer in self.downlayer_8:
            out = layer(out)
        d8 = out
        
        for layer in self.downlayer_32:
            out = layer(out)
        d32 = out
        _,_,H,W = out.shape
        #print 'forward 2', out.shape
        out = mx.nd.contrib.BilinearResize2D(out,height=H*4,width=W*4)
        out = self.uplayer_8(out)
        out = out + d8
        _,_,H,W = out.shape
        
        out = mx.nd.contrib.BilinearResize2D(out,height=H*4,width=W*4)
        out = self.uplayer_2(out)
        out = out  + d2
        _,_,H,W = out.shape
        out = mx.nd.contrib.BilinearResize2D(out,height=H*2,width=W*2)
        out = self.uplayer_1(out)
        

        return out

def get_net(num_class,root):
    return FCN(num_class,root=root)

if 0:
    ctx = mx.gpu()
    fcn = FCN(2)
    fcn.collect_params().reset_ctx(ctx)
    x = mx.nd.random.uniform(0,1,(2,3,64,64),ctx=ctx)
    print 'input: ',x.shape
    y = fcn(x)
    print 'output: ',y.shape

