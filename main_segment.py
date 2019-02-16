import mxnet as mx
from mxnet.gluon import Trainer
from utils import CycleScheduler,FocusLoss
from datasets import thread,segment_voc
from networks import fcn,enet
from utils import train_seg
import os



ctx = mx.gpu(0)
batch_size = 10
num_epochs = 5000
base_lr = 0.0001 #should be small for model with pretrained model
wd = 0.0005
net_name = "enet"
dataset_name = 'voc'
load_to_train = False
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

if load_to_train:
    net.load_parameters('output/enet.params')

net.collect_params().reset_ctx(ctx)

iter_per_epoch = num_train // batch_size
lr_sch = CycleScheduler(updates_one_cycle = iter_per_epoch * 5, min_lr = base_lr/10, max_lr = base_lr * 5)


trainer = Trainer(net.collect_params(),optimizer="sgd",optimizer_params={"wd":wd})

loss = FocusLoss(axis=1,gamma=2)

train_seg(net,train_iter,test_iter,batch_size,trainer,ctx, num_epochs, lr_sch, save_prefix=output_prefix, cls_loss=loss)

