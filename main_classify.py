import mxnet as mx
from mxnet.gluon import Trainer
from mxnet import lr_scheduler
from datasets import fasionmnist,classify_dataset,cifar,cub200
from networks import alexnet,vgg,nin,googlelenet,resnet,densenet,squeezenet
from utils import train_net
from tools import lr_schs
import os,pdb

class NET_DATASET_CONFIG(object):
    def __init__(self):
        return
    def trainer(self,net):
        return None
    def learning_rate_scheduler(self):
        return None
    def epoch_num(self):
        return 0
    def batch_size(self):
        return 0

class RESNET_CIFAR10_CONFIG(NET_DATASET_CONFIG):
    def __init__(self):
        self._num_epochs = 150
        self._batch_size = 128
        self._base_lr = 0.1
    def trainer(self,net):
        print("trainer \t optim:sgd wd: 1e-4, momentum:0.9")
        return Trainer(net.collect_params(),optimizer="sgd",optimizer_params={"wd":1e-4,"momentum":0.9})
    def learning_rate_scheduler(self):
        print("learning_rate \t base_lr:%d, scheduler: MultiFactorScheduler"%self._base_lr)
        lr_sch = lr_scheduler.MultiFactorScheduler(step=[int(self._num_epochs * 0.5), int(self._num_epochs * 0.75) ], factor=0.1)
        lr_sch.base_lr = self._base_lr
        return lr_sch
    def epoch_num(self):
        print("epoch: %d"%self._num_epochs)
        return self._num_epochs
    def batch_size(self):
        print('batch size: %d'%self._batch_size)
        return self._batch_size


class RESNET_CUB200_CONFIG(NET_DATASET_CONFIG):
    def __init__(self):
        self._num_epochs = 150
        self._batch_size = 64//4
        self._base_lr = 0.1
    def trainer(self,net):
        print("trainer \t optim:sgd wd: 1e-4, momentum:0.9")
        return Trainer(net.collect_params(),optimizer="sgd",optimizer_params={"wd":1e-4,"momentum":0.9})
    def learning_rate_scheduler(self):
        print("learning_rate \t base_lr:%.4f, scheduler: MultiFactorScheduler"%self._base_lr)
        lr_sch = lr_scheduler.MultiFactorScheduler(step=[int(self._num_epochs * 0.5), int(self._num_epochs * 0.75) ], factor=0.1)
        lr_sch.base_lr = self._base_lr
        return lr_sch
    def epoch_num(self):
        print("epoch: %d"%self._num_epochs)
        return self._num_epochs
    def batch_size(self):
        print('batch size: %d'%self._batch_size)
        return self._batch_size

class VGG_CIFAR10_CONFIG(NET_DATASET_CONFIG):
    def __init__(self):
        self._num_epochs = 100
        self._batch_size = 128
        self._base_lr = 0.1
    def trainer(self, net):
        wd, mom = 1e-4,0.9
        print("trainer \t optim:sgd wd: %f, momentum: %f"%(wd,mom))
        return Trainer(net.collect_params(), optimizer="sgd", optimizer_params={"wd": wd, "momentum": mom})

    def learning_rate_scheduler(self):
        warmup = 10
        print("learning_rate \t base_lr:%d, scheduler: CosineScheduler with warmup %d" % (self._base_lr,warmup))
        lr_sch = lr_schs.CosineScheduler(num_epochs, self._base_lr,warmup=warmup)
        #lr_sch.base_lr = self._base_lr
        return lr_sch

    def epoch_num(self):
        print("epoch: %d" % self._num_epochs)
        return self._num_epochs

    def batch_size(self):
        print('batch size: %d' % self._batch_size)
        return self._batch_size


configs = {
    "vgg-11,cifar":VGG_CIFAR10_CONFIG(),
    "resnet,cifar":RESNET_CIFAR10_CONFIG(),
    "resnet,cub200":RESNET_CUB200_CONFIG()
}


if __name__=="__main__":
    config_name = "resnet,cub200"
    ctx = mx.gpu(0)
    batch_size = configs[config_name].batch_size()
    num_epochs = configs[config_name].epoch_num()
    net_name, data_name = config_name.split(',')
    output_folder = os.path.join("output")
    output_prefix = os.path.join(output_folder,net_name)


    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
  


    if data_name == "classify_dataset":
        train_iter,test_iter, num_train = classify_dataset.load(batch_size,None)
        class_names = classify_dataset.get_class_names()
    elif data_name == "fasionmnist":
        train_iter,test_iter, num_train = fasionmnist.load(batch_size,resize=None)
        class_names = fasionmnist.get_class_names()
    elif data_name == "cifar":
        if net_name == "squeezenet":
            batch_size = 128*2 #squeezenet requires large batch_size
            print('update batch size to ', batch_size)
            train_iter,test_iter,class_names = cifar.load(batch_size,None)
        else:
            train_iter,test_iter,class_names = cifar.load(batch_size)
    elif data_name == "cub200":
        train_iter,test_iter,class_names = cub200.load(batch_size)
    else:
        rec_train,rec_test = "fortrain.rec", "fortest.rec"  
        train_iter = mx.io.ImageRecordIter( path_imgrec = rec_train, data_shape = (3,32,32), batch_size = batch_size )
        test_iter = mx.io.ImageRecordIter( path_imgrec = rec_test, data_shape = (3,32,32), batch_size = batch_size )
        class_names = '0,1,2,3,4,5,6,7,8,9'.split(',')
        num_train = 50000
        print("class names:",class_names)
        print("# train images:",num_train)
        
        

    if net_name == "alexnet":
        net = alexnet.load(len(class_names),fc_size = 256)
    elif net_name == "vgg-11":
        net = vgg.load("vgg-11",len(class_names))
    elif net_name == "nin":
        net = nin.load(len(class_names))
        base_lr = 0.1
    elif net_name == "googlelenet":
        net = googlelenet.load(len(class_names))
        base_lr = 0.1
        resize=(96,96)
    elif (net_name == "resnet-N" or net_name == "resnet"):
        net = resnet.load('resnet-N',len(class_names))
    elif net_name == "resnet-164":
        net = resnet.load('resnet-164',len(class_names))
    elif net_name == "densenet":
        net = densenet.load(len(class_names))
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


    net.collect_params().reset_ctx(ctx)
    trainer = configs[config_name].trainer(net)
    lr_sch = configs[config_name].learning_rate_scheduler()




    train_net(net,train_iter,test_iter,batch_size,trainer,ctx, num_epochs, lr_sch, output_prefix)

