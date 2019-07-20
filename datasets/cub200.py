import mxnet as mx
from mxnet import gluon
import cv2,random,os
import numpy as np

class CUB200(mx.gluon.data.Dataset):
    def __init__(self,fortrain,root=None,resize=None):
        super(CUB200,self).__init__()
        if root is None:
            root = os.path.join(os.getenv("ENV_DATASET_DIR"), "CUB_200_2011")
        images = []
        with open(os.path.join(root,"images.txt"),'r') as f:
            for line in f:
                image_id,relative_path = line.strip().split(' ')
                images.append((relative_path,int(image_id)))
        classes = []
        with open(os.path.join(root,"classes.txt"),'r') as f:
            for line in f:
                class_id, class_name = line.strip().split(' ')
                classes.append((class_name, int(class_id)-1))
        classes = dict(classes)

        flags = []
        with open(os.path.join(root,"train_test_split.txt"),'r') as f:
            for line in f:
                image_id, flag = [int(x) for x in line.strip().split(' ')]
                if not fortrain:
                    flag = 1 - flag
                flags.append( (image_id, flag)   )
        flags = dict(flags)

        self._mean = np.array([0.4914, 0.4822, 0.4465])
        self._std = np.array([0.2023, 0.1994, 0.2010])

        self._fortrain = fortrain
        self.class_names = classes.keys()
        self.data_list = []
        for path,image_id in images:
            class_name = path.strip().split('/')[0]
            class_id = classes[class_name]
            if flags[image_id]:
                self.data_list.append((os.path.join(root,"images",path),  class_id))
        return

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self,idx):
        image = cv2.imread(self.data_list[idx][0],1)
        label = int(float(self.data_list[idx][1]))
        image = np.float32(image) / 255.0
        image = mx.nd.array(image)

        data_shape = (3,244,244)
        if self._fortrain:
            augs = mx.image.CreateAugmenter(data_shape=data_shape, resize=0, rand_mirror=True, rand_crop=True,
             #                            brightness = 0.5,contrast = 0.5, 
                                         #saturation  = 0.5, hue = 0.5,
                                         mean=self._mean, std=self._std)
        else:
            augs = mx.image.CreateAugmenter(data_shape=data_shape, resize=0, rand_mirror=False, rand_crop=False,
                                         mean=self._mean, std=self._std)

        for aug in augs:
            image = aug(image)

        if 0:
            vis = image.asnumpy() * self._std + self._mean
            vis = vis * 255
            print(vis.max(), vis.min())
            vis = np.clip(vis,0,255).astype(np.uint8)
            cv2.imshow("vis",vis)
            cv2.waitKey(-1)

        image = mx.nd.transpose(image,(2,0,1))
        return image,label

    def get_class_names(self):
        return self.class_names

def load(batch_size):
    trainset = CUB200(True)
    testset = CUB200(False)
    train_iter = gluon.data.DataLoader(trainset,batch_size,shuffle=True,last_batch="rollover",num_workers=3)
    test_iter = gluon.data.DataLoader(testset,batch_size,shuffle=False,last_batch="rollover",num_workers=3)
    print('cub200: train {} test {}'.format(len(trainset), len(testset)))
    return train_iter, test_iter, trainset.get_class_names()

if 0:
    train_iter, _, names = load(10)
    print(names)
    for Y,X in train_iter:
        print(Y.shape, X.shape)
