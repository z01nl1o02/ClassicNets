import mxnet as mx
from mxnet import gluon
from utils import SpatialDropout2D


show_log = False

class ENET_CONV(gluon.Block):
    def __init__(self,channels,kernel_size,conv_type="normal", dilation=1, strides=(1,1)):
        super(ENET_CONV,self).__init__()
        self.conv_type = conv_type
        if conv_type == "dilated":
            self.conv = gluon.nn.Conv2D(channels=channels,kernel_size=kernel_size, dilation=dilation, strides=strides,
                                        padding = kernel_size * dilation//2, use_bias=False)
        else:
            self.conv = gluon.nn.Conv2D(channels=channels,kernel_size=kernel_size, strides=strides,
                                        padding = kernel_size//2, use_bias=False)
        self.prelu = gluon.nn.PReLU()
        self.bn = gluon.nn.BatchNorm()

    def forward(self, x):
        if self.conv_type == "upsampling":
            _,_,H,W = x.shape
            x = mx.nd.contrib.BilinearResize2D(x, height=H * 2, width=W * 2)
        out = self.prelu(self.bn(self.conv(x)))
        if show_log:
            print 'ENET_CONV ({})'.format(self.conv_type),self.name, x.shape, out.shape
        return out

class ENET_INIT(gluon.nn.Block):
    def __init__(self):
        super(ENET_INIT,self).__init__()
        self.maxpool = gluon.nn.MaxPool2D((2,2),strides=2,padding=0)
        self.conv = ENET_CONV(13,3,strides=2)
        return
    def forward(self, x):
        x1 = self.maxpool(x)
        x2 = self.conv(x)
        if show_log:
            print self.name,x.shape,x1.shape,x2.shape
        return mx.nd.concat(x1,x2,dim=1)

class ENET_BOTTLENECK(gluon.nn.Block):
    def __init__(self,channels,downsampling=False,conv_type = "normal",kernel_size=(3,3)):
        super(ENET_BOTTLENECK,self).__init__()
        self.downsampling = downsampling
        if downsampling:
            self.b1 = gluon.nn.Sequential()
            self.b1.add(
                ENET_CONV(channels=channels,kernel_size=3,strides=2),
                ENET_CONV(conv_type=conv_type,channels=channels,kernel_size=kernel_size,strides=1),
                ENET_CONV(channels=channels,kernel_size=1,strides=1)
            )
            self.regular = SpatialDropout2D(0.5)
            self.b2 = gluon.nn.Sequential()
            self.b2.add(
                gluon.nn.MaxPool2D(pool_size=(2,2), strides=2,padding=0),
                gluon.nn.Conv2D(channels=channels,kernel_size=1,strides=1,padding=0)
            )
            self.conv = gluon.nn.Conv2D(channels=channels,kernel_size=1,strides=1,padding=0)
            self.prelu = gluon.nn.PReLU()
        else:
            self.b1 = gluon.nn.Sequential()
            self.b1.add(
                ENET_CONV(channels=channels,kernel_size=1,strides=1),
                ENET_CONV(conv_type=conv_type,channels=channels,kernel_size=kernel_size,strides=1),
                ENET_CONV(channels=channels,kernel_size=1,strides=1)
            )


        return
    def forward(self,x):
        x1 = x
        for layer in self.b1:
            x1 = layer(x1)
            if show_log:
                print 'ENET_BOTTLENECK:',layer.name, x1.shape
        if self.downsampling:
            x1 = self.regular(x1)
            x2 = self.b2(x)
            return self.prelu( self.conv(x1 + x2) )
        return x1


class ENET(gluon.Block):
    def __init__(self, num_class):
        super(ENET,self).__init__()
        self.stages = gluon.nn.Sequential()
        self.stages.add(
            ENET_INIT(),

            ENET_BOTTLENECK(64,downsampling=True,kernel_size=3),
            ENET_BOTTLENECK(64,downsampling=False,kernel_size=3),
            ENET_BOTTLENECK(64,downsampling=False,kernel_size=3),
            ENET_BOTTLENECK(64,downsampling=False,kernel_size=3),
            ENET_BOTTLENECK(64,downsampling=False,kernel_size=3),


            ENET_BOTTLENECK(128,downsampling=True,kernel_size=3),

            ENET_BOTTLENECK(128,downsampling=False,kernel_size=3),
            ENET_BOTTLENECK(128,downsampling=False,kernel_size=3,conv_type="dilated"), #2
            ENET_BOTTLENECK(128,downsampling=False,kernel_size=5,conv_type="asymmetric"),
            ENET_BOTTLENECK(128,downsampling=False,kernel_size=3,conv_type="dilated"), #4
            ENET_BOTTLENECK(128,downsampling=False,kernel_size=3),
            ENET_BOTTLENECK(128,downsampling=False,kernel_size=3,conv_type="dilated"), #8
            ENET_BOTTLENECK(128,downsampling=False,kernel_size=5,conv_type="asymmetric"),
            ENET_BOTTLENECK(128,downsampling=False,kernel_size=3,conv_type="dilated"), #16

            ENET_BOTTLENECK(128,downsampling=False,kernel_size=3),
            ENET_BOTTLENECK(128,downsampling=False,kernel_size=3,conv_type="dilated"), #2
            ENET_BOTTLENECK(128,downsampling=False,kernel_size=5,conv_type="asymmetric"),
            ENET_BOTTLENECK(128,downsampling=False,kernel_size=3,conv_type="dilated"), #4
            ENET_BOTTLENECK(128,downsampling=False,kernel_size=3),
            ENET_BOTTLENECK(128,downsampling=False,kernel_size=3,conv_type="dilated"), #8
            ENET_BOTTLENECK(128,downsampling=False,kernel_size=5,conv_type="asymmetric"),
            ENET_BOTTLENECK(128,downsampling=False,kernel_size=3,conv_type="dilated"), #16


            ENET_BOTTLENECK(64,kernel_size=3,conv_type="upsampling"),
            ENET_BOTTLENECK(64,kernel_size=3),
            ENET_BOTTLENECK(64,kernel_size=3),

            ENET_BOTTLENECK(16,kernel_size=3,conv_type="upsampling"),
            ENET_BOTTLENECK(16,kernel_size=3),
        )
        self.lastConv = gluon.nn.Conv2D(num_class,3,strides=1,padding=1,use_bias=False)

        self.stages.initialize()
        self.lastConv.initialize()
        return

    def forward(self, x):
        out = self.stages(x)
        _,_,H,W = out.shape
        out = mx.nd.contrib.BilinearResize2D(out,height=H*2,width=W*2)
        out = self.lastConv(out)
        return out

def get_net(num_class):
    return ENET(num_class)


if 0:
    ctx = mx.gpu()
    net = ENET(21)
    #for name in net.collect_params('.*weight'):
    #    print name,net.collect_params()[name].shape
        #print name, param.shape
    net.collect_params().reset_ctx(ctx)
    x = mx.nd.random.uniform(0, 1, (2, 3, 512, 512), ctx=ctx)
    print 'input: ', x.shape
    y = net(x).asnumpy()
    #print y.shape, y.min(), y.max()
    print 'output: ', y.shape

