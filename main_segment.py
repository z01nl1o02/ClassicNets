import mxnet as mx
from mxnet.gluon import Trainer
from utils import FocusLoss,WeightCELoss
from datasets import thread,segment_voc,segment_vocaug,segment_voc_human
from networks import fcn,enet,unet
from utils import train_seg,MIOU
import os


if __name__=="__main__":

    ctx = mx.gpu(0)
    batch_size = 16
    num_epochs = 100
    base_lr = 0.001 #should be small for model with pretrained model
    wd = 0.0005
    net_name = "fcn" #"unet"
    dataset_name = 'segment_voc_human'
    label_scale = 1 #8 4 2 1     #enet train from raw to fine
    #load_to_train = False
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
    elif dataset_name == "segment_voc_human":
        class_names = segment_voc_human.get_class_names()
        train_iter, test_iter, num_train = segment_voc_human.load(batch_size,scale=label_scale)
    elif dataset_name == "vocaug":
        class_names = segment_vocaug.get_class_names()
        train_iter, test_iter, num_train = segment_vocaug.load(batch_size,scale=label_scale)

    if net_name == "fcn":
        net = fcn.get_net(len(class_names))
    elif net_name == "enet":
        net = enet.get_net(len(class_names),label_downscale=label_scale)
    elif net_name == "unet":
        net = unet.get_net(len(class_names))

    if os.path.exists('output/unet.params'):
        print('finetuning based on pretrained model')
        net.load_parameters('output/unet.params',allow_missing=True,ignore_extra=True)

    #for key in net.collect_params():
    #    print(key)

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
    #weight_class[0] = 0.01
    loss = WeightCELoss(axis=1, weight_classes = weight_class)


    mIoU = MIOU(class_names)

    train_seg(net,train_iter,test_iter,batch_size,trainer,ctx, num_epochs, lr_sch, save_prefix=output_prefix, cls_acc = mIoU, cls_loss=loss)

