import mxnet as mx
from mxnet.gluon import Trainer
from utils import CycleScheduler,FocusLoss,WeightCELoss
from datasets import thread,segment_voc
from networks import fcn,enet
from utils import train_seg
import os



ctx = mx.gpu(0)
batch_size = 3 #32
num_epochs = 1000
base_lr = 0.001 #should be small for model with pretrained model
wd = 0.0005
net_name = "enet"
dataset_name = 'voc'
label_scale = 2 #8 4 2 1     #enet train from raw to fine
load_to_train = True
output_folder = os.path.join("output")
output_prefix = os.path.join(output_folder,net_name+"_")


if not os.path.exists(output_folder):
    os.makedirs(output_folder)

    
if dataset_name == "thread":
    class_names = thread.get_class_names()
    train_iter, test_iter, num_train = thread.load(batch_size)
elif dataset_name == 'voc':
    class_names = segment_voc.get_class_names()
    train_iter, test_iter, num_train = segment_voc.load(batch_size,scale=label_scale)

if net_name == "fcn":
    net = fcn.get_net(len(class_names),root='networks/')
elif net_name == "enet":
    
    net = enet.get_net(len(class_names),scale=label_scale)

if load_to_train:
    net.load_parameters('output/enet.params',allow_missing=True,ignore_extra=True)
    

print("train set size = ",num_train)

net.collect_params().reset_ctx(ctx)

iter_per_epoch = num_train // batch_size
#lr_sch = CycleScheduler(updates_one_cycle = iter_per_epoch * 5, min_lr = base_lr/10, max_lr = base_lr * 5)
lr_sch = mx.lr_scheduler.PolyScheduler(max_update= iter_per_epoch * num_epochs,base_lr=base_lr, pwr=1)
#lr_sch = mx.lr_scheduler.FactorScheduler(num_epochs *  iter_per_epoch//3,factor=0.1,stop_factor_lr=1e-8)
#lr_sch.base_lr = base_lr

trainer = Trainer(net.collect_params(),optimizer="sgd",optimizer_params={"wd":wd})

#loss = FocusLoss(axis=1,gamma=1)
weight_class = [1.0 for k in range(len(class_names))]
weight_class[0] = 0.01
loss = WeightCELoss(axis=1, weight_classes = weight_class)

train_seg(net,train_iter,test_iter,batch_size,trainer,ctx, num_epochs, lr_sch, save_prefix=output_prefix, cls_loss=loss)

