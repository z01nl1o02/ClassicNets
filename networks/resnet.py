from mxnet.gluon import nn
from mxnet import nd
import mxnet as mx

class Residual(nn.Block):
    def __init__(self, num_channels, use_1x1conv=False, strides=1, **kwargs):
        super(Residual,self).__init__(**kwargs)
        self.conv1 = nn.Conv2D(num_channels, kernel_size=3, strides=strides,padding=1,use_bias=False)
        self.conv2 = nn.Conv2D(num_channels, kernel_size=3, padding=1,use_bias=False)
        if use_1x1conv:
            self.conv3 = nn.Conv2D(num_channels,kernel_size=1,strides=strides,use_bias=False)
            self.bn3 = nn.BatchNorm()
        else:
            self.conv3 = None
            self.bn3 = None
        self.bn1 = nn.BatchNorm()
        self.bn2 = nn.BatchNorm()

    def forward(self, X):
        Y = nd.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.bn3(self.conv3(X))
        return nd.relu(Y + X)

def resnet_block(num_channels, num_residuals, first_block=False):
    blk = nn.Sequential()
    for i in range(num_residuals):
        if i == 0 and not first_block: #first resnet block not resize  and other block resize to 0.5x
            blk.add(
                Residual(num_channels, use_1x1conv=True, strides=2)
            )
        else:
            blk.add(
                Residual(num_channels)
            )
    return blk
    
def resnetN(num_classes,N=3):
    net = nn.Sequential()
    net.add(
        nn.Conv2D(16,kernel_size=3,strides=1,padding=1),
        nn.BatchNorm(), nn.Activation("relu"))
    net.add( resnet_block(16,N,True), resnet_block(32,N), resnet_block(64,N) )
    net.add( nn.GlobalAvgPool2D(), nn.Dense(num_classes))
    return net
        

def resnet_18(num_classes):
    net = nn.Sequential()
    net.add(
        nn.Conv2D(64,kernel_size=7,strides=2,padding=3),
        nn.BatchNorm(), nn.Activation("relu"),
        nn.MaxPool2D(pool_size=3, strides=2, padding=1)
    )
    net.add(
        resnet_block(64, 2, first_block=True),
        resnet_block(128, 2),
        resnet_block(256,2),
        resnet_block(512,2),
        nn.GlobalAvgPool2D(),
        nn.Dense(num_classes)
    )
    return net

    
import mxnet as mx
from mxnet.gluon import nn


class Residual_v2_bottleneck(nn.HybridBlock):
    def __init__(self, channels, same_shape=True):
        super(Residual_v2_bottleneck, self).__init__()
        self.same_shape = same_shape
        with self.name_scope():
            strides = 1 if same_shape else 2
            self.bn1 = nn.BatchNorm()
            self.conv1 = nn.Conv2D(channels // 4, 1, use_bias=False)
            self.bn2 = nn.BatchNorm()
            self.conv2 = nn.Conv2D(
                channels // 4, 3, padding=1, strides=strides, use_bias=False)
            self.bn3 = nn.BatchNorm()
            self.conv3 = nn.Conv2D(channels, 1, use_bias=False)
            self.bn4 = nn.BatchNorm()

            if not same_shape:
                self.conv4 = nn.Conv2D(
                    channels, 1, strides=strides, use_bias=False)

    def hybrid_forward(self, F, x):
        out = self.conv1(self.bn1(x))
        out = F.relu(self.bn2(out))
        out = F.relu(self.bn3(self.conv2(out)))
        out = self.bn4(self.conv3(out))
        if not self.same_shape:
            x = self.conv4(x)
        return out + x


class ResNet164_v2(nn.HybridBlock):
    def __init__(self, num_classes, verbose=False):
        super(ResNet164_v2, self).__init__()
        self.verbose = verbose
        with self.name_scope():
            net = self.net = nn.HybridSequential()
            # block 1
            net.add(nn.Conv2D(64, 3, 1, 1, use_bias=False))
            # block 2
            for _ in range(27):
                net.add(Residual_v2_bottleneck(64))
            # block 3
            net.add(Residual_v2_bottleneck(128, same_shape=False))
            for _ in range(26):
                net.add(Residual_v2_bottleneck(128))
            # block 4
            net.add(Residual_v2_bottleneck(256, same_shape=False))
            for _ in range(26):
                net.add(Residual_v2_bottleneck(256))
            # block 5
            net.add(nn.BatchNorm())
            net.add(nn.Activation('relu'))
            net.add(nn.AvgPool2D(8))
            net.add(nn.Dense(num_classes))

    def hybrid_forward(self, F, x):
        out = x
        for i, b in enumerate(self.net):
            out = b(out)
            if self.verbose:
                print('Block %d output: %s' % (i + 1, out.shape))
        return out    
    
def load(type,num_classes):
    if type == "resnet-18":
        return resnet_18(num_classes)
    if type == "resnet-164":
        #https://github.com/L1aoXingyu/cifar10-gluon/blob/master/resnet.py
        net = ResNet164_v2(num_classes)
        net.initialize(init=mx.initializer.Xavier())
        net.hybridize()
        return net
    if type == "resnet-N":
        #cifar-10
        net = resnetN(num_classes,3)
        net.initialize(init=mx.initializer.Xavier())
        return net
    return None


if 0:
    ctx = mx.gpu()
    X = nd.random.uniform(0,1,(1,3,32,32),ctx=ctx)
    net = load("resnet-164",10)
    #net.initialize(ctx=ctx)
    net.collect_params().reset_ctx(ctx)
    Y = X
    #for layer in net:
    Y = net(Y)
    print('{} shape: {}'.format(X.shape, Y.shape))




