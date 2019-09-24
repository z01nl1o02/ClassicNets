import mxnet as mx
from mxnet import gluon
import gluoncv
import numpy as np
import cv2

#for tiny target batchnorm and dropout may cause much difference between train-stage and test-stage
#this fcn is based on vgg1 without batchnorm/dropout
class FCN(gluon.Block):
    def __init__(self, num_class):
        super(FCN,self).__init__()
        backbone = gluon.model_zoo.vision.vgg11(pretrained=True)
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
            gluon.nn.Conv2D(channels=64,kernel_size=3,strides=1,padding=1,activation="relu"),
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
        #out = out + d8
        out = mx.nd.concat(out,d8,dim=1)
        _,_,H,W = out.shape
        
        out = mx.nd.contrib.BilinearResize2D(out,height=H*4,width=W*4)
        out = self.uplayer_2(out)
        #out = out  + d2
        out = mx.nd.concat(out,d2,dim=1)
        _,_,H,W = out.shape
        out = mx.nd.contrib.BilinearResize2D(out,height=H*2,width=W*2)
        out = self.uplayer_1(out)
        

        return out


class FCN_WITH_VGG(gluon.nn.HybridBlock):
    def __init__(self,num_class,**kwargs):
        super(FCN_WITH_VGG,self).__init__(**kwargs)
        pretrained = gluon.model_zoo.vision.vgg11(pretrained=True)
        if 0:
            data = mx.sym.var("data")
            internals = pretrained(data).get_internals()
            print(internals)
        self.feature_extractor = gluoncv.nn.feature.FeatureExpander(network=pretrained, outputs=['pool4_fwd'], \
                                                                    num_filters=[])
        self.feature_extractor.collect_params().setattr("lr_mult", 0.01)
        self.tail = gluon.nn.HybridSequential()
        self.tail.add(
            gluon.nn.Conv2D(num_class,kernel_size=1,strides=1,padding=0,dilation=1,weight_initializer=mx.init.Xavier()),
            gluon.nn.Activation("sigmoid")
        )
        self.tail.collect_params().initialize()
        return
    def hybrid_forward(self, F, x, *args, **kwargs):
        y = self.feature_extractor(x)
        y = self.tail(y)
        return y

def get_net(num_class):
    return FCN_WITH_VGG(num_class)

if 0:
    ctx = mx.gpu()
    fcn = get_net(4)
    #fcn.initialize()
    fcn.hybridize(static_shape = True, static_alloc = True)
    fcn.collect_params().reset_ctx(ctx)
    x = mx.nd.random.uniform(0,1,(2,3,64*2,64*2),ctx=ctx)
    print('input: ',x.shape)
    y = fcn(x)
    print('output: ',y.shape, ' y range:', y.min(),",",y.max())

