import mxnet as mx
import numpy as np
from mxnet.gluon import Trainer
from mxnet import lr_scheduler,nd
from datasets import detect_voc
from networks import ssd
from utils import predict_ssd
import os,pdb,cv2,sys
from tools.eval_metric import VOC07MApMetric


import logging
logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)

handler = logging.FileHandler("forward_log.txt")
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)


scr_handler = logging.StreamHandler()
scr_handler.setLevel(logging.INFO) 
scr_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(message)s"))


logger.addHandler(handler)
logger.addHandler(scr_handler)

ctx = mx.gpu(0)




testset = detect_voc.DETECT_VOC("test","2007",False)
classes = testset._classes
number_classes = len(classes)


net = ssd.SSD(number_classes)
net.load_parameters('output/ssd.params')
net.hybridize(static_alloc = True, static_shape=True)
#net.hybridize()
net.collect_params().reset_ctx(ctx)


logger.info("========ssd forward===========")
mAP = VOC07MApMetric(class_names=None)
thresh = 0.2

for idx in range(len(testset)):
    feat,target = testset[idx]
    src = testset.get_origin_image_at(idx)
    H,W,C = src.shape
    name = testset.get_name_at(idx)
    X = nd.array( np.expand_dims(feat,0) ).as_in_context(ctx)
    output = predict_ssd(net,X).asnumpy().squeeze()

    inds = np.where( output[:,1] < thresh )[0]
    output = np.delete(output, inds, axis=0)

    for one in output:
        cls,score,x0,y0,x1,y1 = one * np.array([1,1,W,H,W,H])
        x0,x1,y0,y1 = [int(x) for x in [x0,x1,y0,y1]]
        info = "{}_{:.3f}".format(classes[int(cls)],score)
        cv2.putText(src,info,(x0,y0),cv2.FONT_HERSHEY_COMPLEX,1.0,(255,255,0))
        cv2.rectangle(src,(x0,y0),(x1,y1),(255,0,0),2)
    cv2.imwrite(os.path.join('debug',name),src)


    labels,preds = np.expand_dims(target,0), np.expand_dims(output,0)
    labels,preds = np.expand_dims(labels,0), np.expand_dims(preds,0)
    labels,preds = mx.nd.array(labels), mx.nd.array(preds)
    mAP.update(labels,preds)
    if idx > 0 and 0 == (idx % 50):
        logger.info(mAP.get())
        break
#mAP.save_roc_graph(path="debug/")
logger.info("in total:")
logger.info(mAP.get())        
exit(0)
        




