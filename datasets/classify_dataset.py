import mxnet as mx
from mxnet import gluon
import cv2,random
import numpy as np



class CLASSIFY_DATASET(mx.gluon.data.Dataset):
    def __init__(self,listfile,fortrain, resize = None):
        super(CLASSIFY_DATASET,self).__init__()
        self.data_list_ = []
        self.train_flag = fortrain
        self.resize = resize
        with open(listfile,'r') as f:
            for line in f:
                image_path,label = line.strip().split(' ')
                self.data_list_.append((image_path,label))
        return
    def __len__(self):
        return len(self.data_list_)
  
    def __getitem__(self,idx):
        image = cv2.imread(self.data_list_[idx][0],1)
        label = int(float(self.data_list_[idx][1]))

        image = np.float32(image) / 255.0
        image = np.transpose(image,(2,0,1))
        return (image,label)


def get_class_names():
    return '0,1,2,3,4'.split(',')

def load(batch_size,resize):
    trainset = CLASSIFY_DATASET('train.txt',True,resize=resize)
    testset = CLASSIFY_DATASET('test.txt',False,resize=resize)
    train_iter = gluon.data.DataLoader(trainset,batch_size,shuffle=True,last_batch="rollover")
    test_iter = gluon.data.DataLoader(testset,batch_size,shuffle=False,last_batch="rollover")
    return train_iter, test_iter, len(trainset)
