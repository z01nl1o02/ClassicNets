from mxnet.gluon import data as gdata
import os,sys

def load(batch_size, resize = None):
    mnist_train = gdata.vision.FashionMNIST(train=True)
    mnist_test = gdata.vision.FashionMNIST(train=False)

    transformer = []
    if resize:
        transformer.append(gdata.vision.transforms.Resize(resize))
    transformer += [gdata.vision.transforms.ToTensor()]
    transformer = gdata.vision.transforms.Compose(transformer)
    num_worker = 0 if sys.platform.startswith("win32") else 2
    train_iter = gdata.DataLoader(mnist_train.transform_first(transformer),batch_size,shuffle=True,
                                  last_batch="rollover",num_workers=num_worker)
    test_iter = gdata.DataLoader(mnist_test.transform_first(transformer),batch_size,shuffle=False,
                                 last_batch="rollover",num_workers=num_worker)
    return train_iter,test_iter,len(mnist_train)

def get_class_names():
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return text_labels

if 0:
    train_iter,_,_ = load(10,(32,32))
    for batch in train_iter:
        X,Y = batch
        print(X.shape, Y.shape)