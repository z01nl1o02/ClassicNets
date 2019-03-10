import mxnet as mx
from mxnet.gluon import Trainer
from mxnet import lr_scheduler
from datasets import fasionmnist
from networks import alexnet,vgg,nin,googlelenet,resnet,densenet
from utils import train_net,CycleScheduler
import os

ctx = mx.gpu(0)
batch_size = 32
num_epochs = 50
base_lr = 0.01
wd = 0.0005
resize=(32,32)
net_name = "alexnet"
output_folder = os.path.join("output")
output_prefix = os.path.join(output_folder,net_name)


if not os.path.exists(output_folder):
    os.makedirs(output_folder)


class_names = fasionmnist.get_class_names()

if net_name == "alexnet":
    net = alexnet.get_net(len(class_names),kernel_first=3,stride_first=1,padding_first = 1,fc_size=256)
elif net_name == "vgg-11":
    net = vgg.load("vgg-11",len(class_names))
elif net_name == "nin":
    net = nin.load(len(class_names))
    base_lr = 0.1
elif net_name == "googlelenet":
    net = googlelenet.load(len(class_names))
    base_lr = 0.1
    resize=(96,96)
elif net_name == "resnet-18":
    net = resnet.load('resnet-18',len(class_names))
    base_lr = 0.05
    resize=(96,96)
elif net_name == "densenet":
    net = densenet.load(len(class_names))
    base_lr = 0.1
    resize=(96,96)


train_iter,test_iter, num_train = fasionmnist.load(batch_size,resize=resize)


#net.initialize(mx.initializer.Xavier())
net.collect_params().reset_ctx(ctx)

iter_per_epoch = num_train // batch_size
#print iter_per_epoch
#lr_sch = lr_scheduler.FactorScheduler(step=iter_per_epoch * 20, factor=0.1)
#lr_sch.base_lr = base_lr
lr_sch = CycleScheduler(updates_one_cycle=iter_per_epoch*5,min_lr=base_lr/100, max_lr=base_lr)

trainer = Trainer(net.collect_params(),optimizer="sgd",optimizer_params={"wd":wd})


train_net(net,train_iter,test_iter,batch_size,trainer,ctx, num_epochs, lr_sch, output_prefix)

