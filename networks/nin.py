from mxnet.gluon import nn


def nin_block(num_channels, kernel_size,strides, paddings):
    blk = nn.Sequential()
    blk.add(
        nn.Conv2D(num_channels,kernel_size=kernel_size,strides=strides,padding=paddings,activation="relu"),
        nn.Conv2D(num_channels,kernel_size=1,activation="relu"),
        nn.Conv2D(num_channels,kernel_size=1,activation="relu")
    )
    return blk

def load(num_classes):
    net = nn.Sequential()
    net.add(
        nin_block(96,kernel_size=11,strides=4,paddings=0),
        nn.MaxPool2D(pool_size=3, strides=2),
        nin_block(256, kernel_size=5, strides=1,paddings=2),
        nn.MaxPool2D(pool_size=3, strides=2),
        nin_block(384, kernel_size=3,strides=1,paddings=1),
        nn.MaxPool2D(pool_size=3, strides=2),
        nn.Dropout(0.5),
        nin_block(num_classes,kernel_size=3,strides=1,paddings=1),
        nn.GlobalAvgPool2D(),
        nn.Flatten()
    )
    return net