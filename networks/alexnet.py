from mxnet.gluon import nn
import mxnet as mx

class LRN(nn.Block):
    def __init__(self,nsize=5,alpha=0.0001,beta=0.75):
        super(LRN,self).__init__()
        self.nsize = nsize
        self.alpha = alpha
        self.beta = beta
        return
    def forward(self,x):
        return mx.nd.LRN(x,nsize=self.nsize, alpha = self.alpha, beta = self.beta)

class AlexNet(nn.Block):
    def __init__(self,num_classes,  kernel_first = 11, stride_first = 4, padding_first = 0, fc_size = 4096):
        super(AlexNet,self).__init__()
        self.layers = nn.Sequential()
        self.layers.add(
        nn.Conv2D(96,kernel_first,strides=stride_first,padding=padding_first,activation="relu"),
        LRN(),
        nn.MaxPool2D(3,2,padding=1),

        nn.Conv2D(256,5,strides=1,padding=2,activation="relu",groups=2),
        LRN(),
        nn.MaxPool2D(3,2,padding=0),

        nn.Conv2D(384,3,padding=1,activation="relu"),
        nn.Conv2D(384,3,padding=1,activation="relu",groups=2),
        nn.Conv2D(256,kernel_size=3,padding=1,activation="relu",groups=2),
        nn.MaxPool2D(3,2,padding=0),

        nn.Dense(fc_size,activation="relu"), nn.Dropout(0.5),
        nn.Dense(fc_size,activation="relu"), nn.Dropout(0.5),

        nn.Dense(num_classes) )
        return
   # def hybrid_forward(self, F, x):
   #     return self.layers(x)
    def forward(self,x):
        return self.layers(x)

def get_net(num_classes,kernel_first = 11, stride_first = 4, padding_first = 0,fc_size = 4096):
    net = AlexNet(num_classes, kernel_first=kernel_first,stride_first=stride_first,
                  padding_first=padding_first,fc_size=fc_size)
    net.initialize(mx.initializer.Xavier())
    #net.hybridize()
    return net

if 0:
    ctx = mx.gpu(0)
    net = get_net(10)
    net.initialize(mx.initializer.Xavier(),ctx=ctx)
    X = mx.nd.random.uniform(0,1,(32,3,65,65),ctx=ctx)
    X = net(X)
    print("layer {} shape{}".format("alexnet", X.shape))

