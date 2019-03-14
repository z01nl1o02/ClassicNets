from mxnet import nd
import mxnet as mx
from mxnet.gluon import nn

def resnet_conv(num_channels): #updated conv in resnet. run bn before activation
    blk = nn.Sequential()
    blk.add(nn.BatchNorm(),
            nn.Activation('relu'),
            nn.Conv2D(num_channels,kernel_size=3,padding=1)
            )
    return blk

class DenseBlock(nn.Block):
    def __init__(self,num_convs, num_channels, **kwargs):  #input channel: N, output channel: N + num_convs * num_channels
        super(DenseBlock,self).__init__(**kwargs)
        self.net = nn.Sequential()
        for _ in range(num_convs):
            self.net.add(
                resnet_conv(num_channels)
            )
        return
    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            X = nd.concat(X,Y,dim=1) #differenet from resnet here!!!
        return X


def transition_block(num_channels): #reduce channel number
    blk = nn.Sequential()
    blk.add(
        nn.BatchNorm(),nn.Activation("relu"),
        nn.Conv2D(num_channels, kernel_size=1),
        nn.MaxPool2D(pool_size=2,strides=2)
    )
    return blk


def load(num_classes):
    net = nn.Sequential()
    net.add(
        nn.Conv2D(64,kernel_size=7,strides=2,padding=3),
        nn.BatchNorm(), nn.Activation('relu'),
        nn.MaxPool2D(pool_size=3, strides=2,padding=1)
    )
    num_channels, growth_rate = 64, 32
    num_convs_in_dense_blocks = [4,4,4,4]
    for i, num_convs in enumerate(num_convs_in_dense_blocks):
        net.add(
            DenseBlock(num_convs,growth_rate)
        )
        num_channels += num_convs * growth_rate
        if i != len(num_convs_in_dense_blocks) - 1:
            net.add(
                transition_block(num_channels//2)
            )
    net.add(
        nn.BatchNorm(), nn.Activation("relu"),
        nn.GlobalAvgPool2D(),nn.Dense(num_classes)
    )
    return net

if 0:
    ctx = mx.gpu(0)
    X = nd.random.uniform(0,1,(1,3,96,96),ctx=ctx)
    net = load(10)
    net.initialize(ctx=ctx)
    Y = X
    for blk in net:
        Y = blk(Y)
        print('{} shape: {}'.format(blk.name, Y.shape))

