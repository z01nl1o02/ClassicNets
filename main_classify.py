import mxnet as mx
from mxnet.gluon import Trainer
from mxnet import lr_scheduler
from datasets import fasionmnist,classify_dataset,cifar
from networks import alexnet,vgg,nin,googlelenet,resnet,densenet,squeezenet
from utils import train_net
from tools import lr_schs
import os,pdb

if __name__=="__main__":

    ctx = mx.gpu(0)
    batch_size = 32
    num_epochs = 500
    base_lr = 0.0001
    wd = 0.0004
    mom = 0.9
    resize=(32,32)
    #resize = None
    net_name = "squeezenet"
    data_name = "cifar"

    output_folder = os.path.join("output")
    output_prefix = os.path.join(output_folder,net_name)


    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if data_name == "classify_dataset":
        train_iter,test_iter, num_train = classify_dataset.load(batch_size,resize=resize)
        class_names = classify_dataset.get_class_names()
    elif data_name == "fasionmnist":
        train_iter,test_iter, num_train = fasionmnist.load(batch_size,resize=resize)
        class_names = fasionmnist.get_class_names()
    elif data_name == "cifar":
        if net_name == "squeezenet":
	    batch_size = 128*2 #squeezenet requires large batch_size
	    print('update batch size to ', batch_size)
            train_iter,test_iter,class_names = cifar.load(batch_size,resize)
        else:
            train_iter,test_iter,class_names = cifar.load(batch_size)
    else:
        rec_train,rec_test = "fortrain.rec", "fortest.rec"  
        train_iter = mx.io.ImageRecordIter( path_imgrec = rec_train, data_shape = (3,32,32), batch_size = batch_size )
        test_iter = mx.io.ImageRecordIter( path_imgrec = rec_test, data_shape = (3,32,32), batch_size = batch_size )
        class_names = '0,1,2,3,4,5,6,7,8,9'.split(',')
        num_train = 50000
        print("class names:",class_names)
        print("# train images:",num_train)
        
        

    if net_name == "alexnet":
        net = alexnet.get_net(len(class_names),fc_size = 256)
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
    elif net_name == "resnet-164":
        net = resnet.load('resnet-164',len(class_names))
    elif net_name == "densenet":
        net = densenet.load(len(class_names))
        base_lr = 0.1
        resize=(96,96)
    elif net_name == "squeezenet":
        net = squeezenet.load(len(class_names),(96,3,1,1))
        base_lr = 0.01 #must be with small lr

   # print(output_prefix + '.params')
    if os.path.exists( output_prefix + '.params' ):
        net.load_parameters( output_prefix + '.params' )
        print('finetune based on ', output_prefix, '.params')

    #for param in net.collect_params(): 
    #    if net.collect_params()[param].grad_req != "null":
    #        pp = net.collect_params()[param].grad()        
            #print(net.collect_params()[param].grad.values())
           # sw.add_histogram(tag=key, values=value.grads(), global_step=iter_num, bins=1000)

    #net.initialize(mx.initializer.Xavier())
    net.collect_params().reset_ctx(ctx)

    #print iter_per_epoch
    #lr_sch = lr_scheduler.FactorScheduler(step=iter_per_epoch * 20, factor=0.1)
    #lr_sch.base_lr = base_lr

#    lr_sch = lr_scheduler.MultiFactorScheduler(step=[10,int(num_epochs * 0.5), int(num_epochs * 0.75) ], factor=0.1, base_lr = base_lr, warmup_steps = 0)
    #lr_sch = lr_scheduler.MultiFactorScheduler(step=[int(num_epochs * 0.1),int(num_epochs * 0.5), int(num_epochs * 0.75) ], factor=0.1)
    #lr_sch.base_lr = base_lr
    lr_sch = lr_schs.CosineScheduler(num_epochs, base_lr)

#    trainer = Trainer(net.collect_params(),optimizer="adam",optimizer_params={"wd":wd,"momentum":mom})
    trainer = Trainer(net.collect_params(),optimizer="adam",optimizer_params={"wd":wd})

    train_net(net,train_iter,test_iter,batch_size,trainer,ctx, num_epochs, lr_sch, output_prefix)

