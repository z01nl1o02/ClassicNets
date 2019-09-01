from mxnet.gluon import nn
from mxnet import nd
import mxnet as mx


class Inception_block(nn.Block):
    def __init__(self,c1,c2,c3,c4, **kwargs):
        super(Inception_block, self).__init__(**kwargs)
        self.p1_1 = nn.Conv2D(c1, kernel_size=1, activation="relu")

        self.p2_1 = nn.Conv2D(c2[0],kernel_size=1, activation="relu")
        self.p2_2 = nn.Conv2D(c2[1],kernel_size=3, padding=1, activation="relu")

        self.p3_1 = nn.Conv2D(c3[0], kernel_size=1,activation="relu")
        self.p3_2 = nn.Conv2D(c3[1],kernel_size=5, padding=2, activation="relu")

        self.p4_1 = nn.MaxPool2D(pool_size=3, strides=1,padding=1)
        self.p4_2 = nn.Conv2D(c4, kernel_size=1, activation="relu")

    def forward(self, x):
        p1 = self.p1_1(x)
        p2 = self.p2_2(self.p2_1(x))
        p3 = self.p3_2(self.p3_1(x))
        p4 = self.p4_2(self.p4_1(x))
        return nd.concat(p1,p2,p3,p4,dim=1)



def load(num_classes):
    b1 = nn.Sequential()
    b1.add(
        nn.Conv2D(64,kernel_size=7,strides=2,padding=3,activation="relu"),
        nn.MaxPool2D(pool_size=3, strides=2, padding=1)
    )
    b2 = nn.Sequential()
    b2.add(
        nn.Conv2D(64,kernel_size=1),
        nn.Conv2D(192,kernel_size=3,padding=1),
        nn.MaxPool2D(pool_size=3, strides=2, padding=1)
    )
    b3 = nn.Sequential()
    b3.add(
        Inception_block(64,(96,128),(16,32),32),
        Inception_block(128,(128,192),(32,96),64),
        nn.MaxPool2D(pool_size=3, strides=2, padding=1)
    )

    b4 = nn.Sequential()
    b4.add(
        Inception_block(192,(96,208),(16,48),64),
        Inception_block(160,(112,224),(24,64),64),
        Inception_block(128,(128,256),(24,64),64),
        Inception_block(112,(144,228),(32,64),64),
        Inception_block(256,(160,320),(32,128),128),
        nn.MaxPool2D(pool_size=3, strides=2, padding=1)
    )

    b5 = nn.Sequential()
    b5.add(
        Inception_block(256, (160,320), (32,128),128),
        Inception_block(384, (192,384),(48,128),128),
        nn.GlobalAvgPool2D()
    )

    net = nn.Sequential()
    net.add(
        b1,b2,b3,b4,b5, nn.Dense(num_classes)
    )
    return net



if 0:
    ctx = mx.gpu(0)
    net = load(10)
    net.initialize(ctx=ctx)
    X = nd.random.uniform(0,1,(1,3,96,96),ctx=ctx)
    for layer in net:
        X = layer(X)
        print("layer {} shape{}".format(layer.name, X.shape))


