import mxnet as mx
import numpy as np
from mxnet.gluon import Trainer
from mxnet import lr_scheduler,nd
from datasets import detect_voc
from networks import ssd
from utils import train_ssd,CycleScheduler,predict_ssd
import os,pdb,cv2

ctx = mx.gpu(0)
batch_size = 32/2
num_epochs = 300
base_lr = 0.004
wd = 0.0005
momentum = 0.9

pretrained = ""#'output/ssd.params'


output_folder = os.path.join("output")
output_prefix = os.path.join(output_folder,"ssd")


if not os.path.exists(output_folder):
    os.makedirs(output_folder)


train_iter,test_iter, classes = detect_voc.load("2007_2012",batch_size)

number_classes = len(classes)

net = ssd.SSD(number_classes)

if pretrained != "":
    net.load_parameters(pretrained)
    print('finetune based on ',pretrained)

net.collect_params().reset_ctx(ctx)


#lr_sch = lr_scheduler.MultiFactorScheduler(step=[int(num_epochs * 0.45), int(num_epochs * 0.7) ], factor=0.1, base_lr = base_lr, warmup_steps = 0)
lr_sch = lr_scheduler.MultiFactorScheduler(step=[int(num_epochs * 0.45), int(num_epochs * 0.7) ], factor=0.1)
lr_sch.base_lr = base_lr

trainer = Trainer(net.collect_params(),optimizer="sgd",optimizer_params={"wd":wd,"momentum":momentum})


train_ssd(net,train_iter,test_iter,batch_size,trainer,ctx, num_epochs, lr_sch, output_prefix)



