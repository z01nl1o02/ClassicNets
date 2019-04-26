from mxnet.gluon import data as gdata
import os,sys
import numpy as np
from mxnet.gluon.data.vision import ImageFolderDataset
from mxnet import nd,image
from mxnet.gluon.data import DataLoader
def get_class_names():
    text_labels = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck'] #error order??
    return text_labels
    

    
def transform_train(data, label):
    im = data.asnumpy()
    im = np.pad(im, ((4, 4), (4, 4), (0, 0)), mode='constant', constant_values=0)
    im = nd.array(im, dtype='float32') / 255
    auglist = image.CreateAugmenter(data_shape=(3, 32, 32), resize=0, rand_mirror=True,
                                    rand_crop=True,
                                   mean=np.array([0.4914, 0.4822, 0.4465]),
                                   std=np.array([0.2023, 0.1994, 0.2010]))
    for aug in auglist:
        im = aug(im)
    im = nd.transpose(im, (2, 0, 1)) # channel x width x height
    return im, nd.array([label]).astype('float32')

def transform_test(data, label):
    im = data.astype('float32') / 255
    auglist = image.CreateAugmenter(data_shape=(3, 32, 32), mean=np.array([0.4914, 0.4822, 0.4465]),
                                   std=np.array([0.2023, 0.1994, 0.2010]))
    for aug in auglist:
        im = aug(im)
    im = nd.transpose(im, (2, 0, 1))
    return im, nd.array([label]).astype('float32')
    
    
    
def transform_train_64(data, label):
    im = data.asnumpy()
    im = np.pad(im, ((4, 4), (4, 4), (0, 0)), mode='constant', constant_values=0)
    im = nd.array(im, dtype='float32') / 255
    auglist = image.CreateAugmenter(data_shape=(3, 36, 36), resize=40, rand_mirror=True,
                                    rand_crop=True,
                                   mean=np.array([0.4914, 0.4822, 0.4465]),
                                   std=np.array([0.2023, 0.1994, 0.2010]))
    for aug in auglist:
        im = aug(im)
    im = nd.transpose(im, (2, 0, 1)) # channel x width x height
    return im, nd.array([label]).astype('float32')

def transform_test_64(data, label):
    im = data.astype('float32') / 255
    auglist = image.CreateAugmenter(data_shape=(3, 36, 36), resize=40,mean=np.array([0.4914, 0.4822, 0.4465]),
                                   std=np.array([0.2023, 0.1994, 0.2010]))
    for aug in auglist:
        im = aug(im)
    im = nd.transpose(im, (2, 0, 1))
    return im, nd.array([label]).astype('float32')    
    
   
def load(batch_size,flag_large = False):
    dataset_dir = os.getenv("ENV_DATASET_DIR")   
    if not flag_large:
        train_ds = ImageFolderDataset(os.path.join(dataset_dir,"cifar","train"), transform=transform_train)
        test_ds = ImageFolderDataset(os.path.join(dataset_dir,"cifar","test"), transform=transform_test)
    else:
        train_ds = ImageFolderDataset(os.path.join(dataset_dir,"cifar","train"), transform=transform_train_64)
        test_ds = ImageFolderDataset(os.path.join(dataset_dir,"cifar","test"), transform=transform_test_64)
    train_data = DataLoader(train_ds, batch_size=batch_size, shuffle=True, last_batch='discard',num_workers = 2)
    test_data = DataLoader(test_ds, batch_size=batch_size, shuffle=False, last_batch='discard',num_workers = 2)
    info = '#train {} #test {}'.format(len(train_ds), len(test_ds))
    return train_data, test_data, info

