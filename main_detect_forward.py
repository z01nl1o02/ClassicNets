import mxnet as mx
import numpy as np
from mxnet.gluon import Trainer
from mxnet import lr_scheduler,nd
from datasets import detect_voc
from networks import ssd
from utils import train_ssd,CycleScheduler,predict_ssd
import os,pdb,cv2
from tools.eval_metric import VOC07MApMetric

ctx = mx.gpu(0)

resize = (256,256)

ds_forward = detect_voc.DETECT_VOC_FORWARD("test","2007")
classes = ds_forward.classes()
number_classes = len(classes)

net = ssd.SSD(number_classes)
import pdb
net.load_parameters('ssd_256x256.params')
net.collect_params().reset_ctx(ctx)


mAP = VOC07MApMetric()

for ind in range(len(ds_forward)):
    feat,target,src,name = ds_forward[ind]
    X = nd.array( np.expand_dims(feat,0) ).as_in_context(ctx)
    output = predict_ssd(net,X).asnumpy()
    H,W,C = src.shape
    for one in output:
        cls,score,x0,y0,x1,y1 = one * np.array([1,1,W,H,W,H])
        if cls >= 0 and score > 0.5:
            x0,x1,y0,y1 = [int(x) for x in [x0,x1,y0,y1]]
            cv2.putText(src,classes[int(cls)],(x0,y0),cv2.FONT_HERSHEY_COMPLEX,1.0,(255,255,0))
            cv2.rectangle(src,(x0,y0),(x1,y1),(255,0,0),2)
    cv2.imwrite(os.path.join('debug',name),src)
    labels,preds = np.expand_dims(target,0), np.expand_dims(output,0)
    labels,preds = np.expand_dims(labels,0), np.expand_dims(preds,0)
    labels,preds = mx.nd.array(labels), mx.nd.array(preds)
    mAP.update(labels,preds)
    if ind > 0 and 0 == (ind % 100):
        print(mAP.get())

print('mAP: ', mAP.get())        
exit(0)
        




