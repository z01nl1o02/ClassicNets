from mxnet.gluon import nn
import mxnet as mx
from mxnet import gluon,nd

def get_net(num_classes):
    net = nn.Sequential()
    net.add(
        nn.Conv2D(96,11,strides=4,activation="relu"),
        nn.MaxPool2D(3,2),

        nn.Conv2D(256,5,strides=1,padding=2,activation="relu"),
        nn.MaxPool2D(3,2),

        nn.Conv2D(384,3,padding=1,activation="relu"),
        nn.Conv2D(384,3,padding=1,activation="relu"),
        nn.Conv2D(256,kernel_size=3,padding=1,activation="relu"),
        nn.MaxPool2D(3,2),

        nn.Dense(4096,activation="relu"), nn.Dropout(0.5),
        nn.Dense(4096,activation="relu"), nn.Dropout(0.5),

        nn.Dense(num_classes) )
    return net


if 0:
    ctx = mx.gpu(0)
    net = get_net(10)
    net.initialize(ctx=ctx)
    X = nd.random.uniform(0,1,(1,3,224,224),ctx=ctx)
    for layer in net:
        X = layer(X)
        print("layer {} shape{}".format(layer.name, X.shape))

