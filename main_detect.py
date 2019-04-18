import mxnet as mx
import numpy as np
from mxnet.gluon import Trainer
from mxnet import lr_scheduler,nd
from datasets import detect_voc
from networks import ssd
from utils import train_ssd,CycleScheduler,predict_ssd
import os,pdb,cv2

ctx = mx.gpu(0)
batch_size = 5
num_epochs = 100
base_lr = 0.1
wd = 0.0001
momentum = 0.9


output_folder = os.path.join("output")
output_prefix = os.path.join(output_folder,"ssd")


if not os.path.exists(output_folder):
    os.makedirs(output_folder)


train_iter,test_iter, number_classes = detect_voc.load("2007",batch_size)


net = ssd.SSD(number_classes)

net.collect_params().reset_ctx(ctx)

if 1:
    net.load_parameters('output/ssd.params')
    image = cv2.imread('000000.jpg',1)
    image = cv2.resize(image,(256,256))
    H,W,C = image.shape
    img = np.float32(image) / 255
    img = np.transpose(img,(2,0,1))
    X = nd.array( np.expand_dims(img,0) ).as_in_context(ctx)
    output = predict_ssd(net,X).asnumpy()
    for one in output:
        cls,score,x0,y0,x1,y1 = one * np.array([1,1,W,H,W,H])
        if cls >= 0 and score > 0.2:
            x0,x1,y0,y1 = [int(x) for x in [x0,x1,y0,y1]]
            cv2.rectangle(image,(x0,y0),(x1,y1),(255,0,0),2)
    cv2.imwrite("out.jpg",image)
    exit(0)
    



lr_sch = lr_scheduler.MultiFactorScheduler(step=[int(num_epochs * 0.45), int(num_epochs * 0.7) ], factor=0.1, base_lr = base_lr, warmup_steps = 0)


trainer = Trainer(net.collect_params(),optimizer="sgd",optimizer_params={"wd":wd,"momentum":momentum})


train_ssd(net,train_iter,test_iter,batch_size,trainer,ctx, num_epochs, lr_sch, output_prefix)



