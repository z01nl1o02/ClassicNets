from mxnet.gluon import nn


def vgg_block(num_convs, num_channels):
    blk = nn.Sequential()
    for _ in range(num_convs):
        blk.add(
            nn.Conv2D(num_channels,kernel_size=3,padding=1,strides=1,activation="relu")
        )
    blk.add(
        nn.MaxPool2D(pool_size=2, strides=2)
    )
    return blk

def vgg_net(conv_arch, num_classes):
    net = nn.Sequential()
    for (num_convs, num_channels) in conv_arch:
        net.add(
            vgg_block(num_convs,num_channels)
        )
    net.add(
        nn.Dense(4096,activation="relu"), nn.Dropout(0.5),
        nn.Dense(4096,activation="relu"), nn.Dropout(0.5),
        nn.Dense(num_classes)
    )
    return net

def load(type,num_classes):
    if type == "vgg-11":
        conv_arch = ((1,46),(1,128),(2,256),(2,512),(2,512))
    else:
        return None
    return vgg_net(conv_arch,num_classes)
