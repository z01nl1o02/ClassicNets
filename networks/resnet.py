from mxnet.gluon import nn
from mxnet import nd
import mxnet as mx

class Residual(nn.Block):
    def __init__(self, num_channels, use_1x1conv=False, strides=1, **kwargs):
        super(Residual,self).__init__(**kwargs)
        self.conv1 = nn.Conv2D(num_channels, kernel_size=3, strides=strides,padding=1)
        self.conv2 = nn.Conv2D(num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2D(num_channels,kernel_size=1,strides=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm()
        self.bn2 = nn.BatchNorm()

    def forward(self, X):
        Y = nd.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
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

def load(type,num_classes):
    if type == "resnet-18":
        return resnet_18(num_classes)
    return None


if 0:
    ctx = mx.gpu()
    X = nd.random.uniform(0,1,(1,3,224,224),ctx=ctx)
    net = load("resnet-18",10)
    net.initialize(ctx=ctx)
    Y = X
    for layer in net:
        Y = layer(Y)
        print('{} shape: {}'.format(layer.name, Y.shape))




