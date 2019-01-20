import mxnet as mx
from mxnet.gluon import Trainer
from mxnet import lr_scheduler
from datasets import thread
from networks import fcn
from utils import train_fcn
import os

ctx = mx.gpu(0)
batch_size = 2
num_epochs = 5000
base_lr = 0.0001 #should be small ! 
wd = 0.0005
net_name = "fcn"
output_folder = os.path.join("output")
output_prefix = os.path.join(output_folder,net_name+"_")


if not os.path.exists(output_folder):
    os.makedirs(output_folder)

class_names = thread.get_class_names()

if net_name == "fcn":
    net = fcn.get_net(len(class_names),root='networks/')


train_iter,test_iter, num_train = thread.load(batch_size)


#net.initialize(mx.initializer.Xavier())
net.collect_params().reset_ctx(ctx)

iter_per_epoch = num_train // batch_size
#print iter_per_epoch
lr_sch = lr_scheduler.FactorScheduler(step=iter_per_epoch * 2500, factor=0.1)
lr_sch.base_lr = base_lr

trainer = Trainer(net.collect_params(),optimizer="sgd",optimizer_params={"wd":wd})


train_fcn(net,train_iter,test_iter,batch_size,trainer,ctx, num_epochs, lr_sch, output_prefix)

