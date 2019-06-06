import os,sys,cv2,pdb
import numpy as np
from mxnet.gluon.data.vision import ImageFolderDataset
from mxnet import nd,image,gluon
from mxnet.gluon.data import DataLoader
########################################################
########################################################

class CIFAR(gluon.data.Dataset):
    def __init__(self, fortrain):
        super(CIFAR,self).__init__()
        self._pad = 4
        self._mean = np.array([0.4914, 0.4822, 0.4465])
        self._std = np.array([0.2023, 0.1994, 0.2010])
        self._classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck'] #error order??
        self._fortrain = fortrain
        self._class_to_ind = dict(zip(self._classes, range(len(self._classes))))
        self._image_exts = '.jpg'.split(',')
        self._data = []
        if self._fortrain:
            root_dir = os.path.join(os.getenv("ENV_DATASET_DIR"), "cifar", "train")
        else:
            root_dir = os.path.join(os.getenv("ENV_DATASET_DIR"), "cifar","test")
        for rdir, pdirs, names in os.walk(root_dir):
            for name in names:
                sname,ext = os.path.splitext(name)
                if ext not in self._image_exts:
                    continue
                path = os.path.join(rdir,name)
                class_name = path.split(os.path.sep)[-2]
                assert(class_name in self._class_to_ind.keys())
                class_id = self._class_to_ind[class_name]
                self._data.append( (path, class_id)  )
        return
    def __len__(self):
        return len(self._data)
    def classes(self):
        return self._classes
    def __getitem__(self,idx):
        path, cid = self._data[idx]
        img = cv2.imread(path,1)

        img = np.pad(img, ((self._pad, self._pad), (self._pad, self._pad), (0, 0)), mode='constant', constant_values=0)
        img = nd.array(img, dtype='float32') / 255
        H,W,C = img.shape
        data_shape = (C,H - self._pad*2, W - self._pad*2)
        if self._fortrain:
            augs = image.CreateAugmenter(data_shape=data_shape, resize=0, rand_mirror=True, rand_crop=True, mean=self._mean, std=self._std)
        else:
            augs = image.CreateAugmenter(data_shape=data_shape,resize=0, rand_mirror=False, rand_crop=False, mean=self._mean, std=self._std)
   
        for aug in augs:
             img = aug(img)

        
        img = nd.transpose(img,(2,0,1))
        return img,nd.array([cid]).astype("float32")
        

def load(batch_size):
    trainset = CIFAR(True)
    testset = CIFAR(False)
    train_iter = gluon.data.DataLoader(trainset, batch_size, shuffle=True, last_batch="rollover",num_workers=3)
    test_iter = gluon.data.DataLoader(trainset, batch_size, shuffle=False, last_batch="rollover",num_workers=2)
    print('cifar: train {} test {}'.format(len(trainset), len(testset)))
    return train_iter, test_iter, trainset.classes()

##################################################################################################
##################################################################################################

if 0:
    print('start...')
    train_iter,_,_ = load(1)
    for X,Y in train_iter:
        Z = X.shape
