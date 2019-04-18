from mxnet.gluon import data as gdata
import os,sys
from mxnet.gluon.data.vision import ImageFolderDataset
from mxnet import nd
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
    
   
def load():
    train_ds = ImageFolderDataset('E:/dataset/train/', transform=transform_train)
    test_ds = ImageFolderDataset('E:/dataset/test/', transform=transform_test)
    train_data = DataLoader(train_ds, batch_size=64, shuffle=True, last_batch='keep')
    test_data = DataLoader(test_ds, batch_size=128, shuffle=False, last_batch='keep')
    info = '#train {} #test {}'.format(len(train_ds), len(test_ds))
    return train_data, test_data, info

