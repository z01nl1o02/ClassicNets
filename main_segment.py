import mxnet as mx
from mxnet.gluon import Trainer
from utils import CycleScheduler
from datasets import thread,segment_voc
from networks import fcn,enet
from utils import train_seg
import os



ctx = mx.gpu(0)
batch_size = 2
num_epochs = 5000
base_lr = 0.001 #should be small ! 
wd = 0.0005
net_name = "enet"
dataset_name = 'voc'
output_folder = os.path.join("output")
output_prefix = os.path.join(output_folder,net_name+"_")


if not os.path.exists(output_folder):
    os.makedirs(output_folder)

if dataset_name == "thread":
    class_names = thread.get_class_names()
    train_iter, test_iter, num_train = thread.load(batch_size)
elif dataset_name == 'voc':
    class_names = segment_voc.get_class_names()
    train_iter, test_iter, num_train = segment_voc.load(batch_size)

if net_name == "fcn":
    net = fcn.get_net(len(class_names),root='networks/')
elif net_name == "enet":
    net = enet.get_net(len(class_names))

net.collect_params().reset_ctx(ctx)

iter_per_epoch = num_train // batch_size
#print iter_per_epoch
#lr_sch = lr_scheduler.FactorScheduler(step=iter_per_epoch * 2500, factor=0.1)
#lr_sch.base_lr = base_lr

lr_sch = CycleScheduler(updates_one_cycle = iter_per_epoch * 2, min_lr = base_lr/10, max_lr = base_lr * 10)


trainer = Trainer(net.collect_params(),optimizer="sgd",optimizer_params={"wd":wd})


train_seg(net,train_iter,test_iter,batch_size,trainer,ctx, num_epochs, lr_sch, output_prefix)

