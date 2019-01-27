import mxnet as mx
from mxnet import gluon
import cv2,random
import numpy as np


#dataset with tiny object 
#so resizeing of large ration may remove target from result
class THREAD(mx.gluon.data.Dataset):
    def __init__(self,fortrain=True,listfile='list.txt'):
        super(THREAD,self).__init__()
        self.data_list_ = []
        self.train_flag = fortrain

        self.shape = (256,320) #output size
        with open(listfile,'rb') as f:
            for line in f:
                image_path = line.strip()
                label_path = image_path.replace("image","mark")
                self.data_list_.append((image_path,label_path))
        return
    def __len__(self):
        return len(self.data_list_)
    def crop_resize(self,image,label):
        if random.randint(0,2) == 1 or not self.train_flag:
            H,W = self.shape
            image, label = cv2.resize(image,(W,H),interpolation=cv2.INTER_AREA), cv2.resize(label,(W,H),interpolation=cv2.INTER_NEAREST)
            return image,label
        H,W,C = image.shape
        h = np.int32(H * 0.9)
        w = np.int32(W * h / H)
        dy,dx = random.randint(0,H-h), random.randint(0,W-w)
        image_crop = image[dy:dy+h, dx:dx+w]
        label_crop = label[dy:dy+h, dx:dx+w]
        H,W = self.shape
        image_crop, label_crop = cv2.resize(image_crop,(W,H),interpolation=cv2.INTER_AREA), cv2.resize(label_crop,(W,H),interpolation=cv2.INTER_NEAREST)
        #print 'crop-resize:',image.shape,label.shape
        return image_crop, label_crop

    def rotation(self,image,label):
        if random.randint(0,2) == 1 or not self.train_flag:
            return image,label
        deg = random.randint(-20, 20)
        #print image.shape
        H, W,C  = image.shape
        center = (H // 2, W // 2)
        M = cv2.getRotationMatrix2D(center, deg, 1.0)
        image = cv2.warpAffine(image, M, (W, H), flags=cv2.INTER_LINEAR)
        label = cv2.warpAffine(label, M, (W, H), flags=cv2.INTER_NEAREST)
        return image,label

    def flip_hor(self,image,label):
        if random.randint(0,2) == 1 or not self.train_flag:
            return image,label
        image = cv2.flip(image,1)
        label = cv2.flip(label,1)
        return image,label

    def flip_ver(self,image,label):
        if random.randint(0,2) == 1 or not self.train_flag:
            return image,label
        image = cv2.flip(image,0)
        label = cv2.flip(label,0)
        return image,label

    def to_tensor(self,image,label):
        #cv2.imshow("image",image)
        #cv2.imshow("label",label)
        #cv2.waitKey(-1)
        to_tensor = gluon.data.vision.transforms.ToTensor()
        label = label[np.newaxis,:,:]
        #print 'to_tensor: ',image.shape, label.shape
        return to_tensor(mx.nd.array(image)), np.int32(label / 255)

    def __getitem__(self,idx):
        image = cv2.imread(self.data_list_[idx][0],1)
        label = cv2.imread(self.data_list_[idx][1],0)

        image,label = self.crop_resize(image,label)
        image,label = self.rotation(image,label)
        image,label = self.flip_hor(image,label)
        image,label = self.flip_ver(image,label)

        image,label = self.to_tensor(image,label)

        return (image,label)


def get_class_names():
    return (0,1)

def load(batch_size):
    trainset = THREAD(fortrain=True)
    testset = THREAD(fortrain=False)
    train_iter = gluon.data.DataLoader(trainset,batch_size,shuffle=True,last_batch="rollover")
    test_iter = gluon.data.DataLoader(testset,batch_size,shuffle=False,last_batch="rollover")
    return train_iter, test_iter, len(trainset)
