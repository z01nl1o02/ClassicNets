from mxnet.gluon import nn
from mxnet import nd
import mxnet as mx
import gluoncv
from gluoncv.model_zoo import get_model


class agg_block(nn.Block):
    def __init__(self, inch, outch):
        super(agg_block,self).__init__()
        with self.name_scope():
            self.downsampling = nn.Sequential()
            self.downsampling.add(
                nn.Conv2D(in_channels=inch, channels=outch, kernel_size=3, padding=1, strides=2),
                nn.BatchNorm(),
                nn.Activation("relu")
            )
            self.bn = nn.BatchNorm()
            self.relu = nn.Activation("relu")
        return
    def forward(self, x0, x1):
        x0 = self.downsampling(x0)
        y = x0 + x1
        y = self.relu(self.bn(y))
        return y

class DLA_IDL(nn.Block):
    def __init__(self, num_classes,pretrained=True):
        super(DLA_IDL,self).__init__()
        model_name = "ResNet50_v2"
        backbone = get_model(model_name,pretrained=pretrained)
        backbone.collect_params().setattr("lr_mult",0.1)

       #for ind,feat in enumerate(backbone.features):
        #    print(feat.name, ind)
        self.output_inds = [5,6,7,10]
        self.backbone = backbone.features[0:11]


        self.agg_blocks = nn.Sequential()
        self.agg_blocks.add(
            agg_block(256,512),
            agg_block(512,1024),
            agg_block(1024,2048)

        )

        self.output = nn.Sequential()
        self.output.add(
            nn.GlobalAvgPool1D(),
            nn.Dense(num_classes),
        )


        self.agg_blocks.initialize(mx.init.Xavier())
        self.output.initialize(mx.init.Xavier())
        return
    def forward(self,x):
        output = []
        for layer in self.backbone:
            x = layer(x)
            output.append(x)
        stage_0 = output[self.output_inds[0]]
        stage_1 = output[self.output_inds[1]]
        stage_2 = output[self.output_inds[2]]
        stage_3 = output[self.output_inds[3]]

        x = self.agg_blocks[0](stage_0, stage_1)
        x = self.agg_blocks[1](x, stage_2)
        x = self.agg_blocks[2](x,stage_3)

        y = self.output(x)

        return y

def load(class_num):
    net = DLA_IDL(class_num)
    return net



if 0:
    ctx = mx.gpu()
    net = resnet_pretrained(10)
    net.collect_params().reset_ctx(ctx)
    x = mx.nd.zeros((2,3,224,224),ctx = ctx)
    y = net(x)
    print(y.shape)
