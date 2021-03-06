from mxnet import gluon
import os,cv2
import numpy as np
import random

def pascal_palette(): #RGB mode
  palette = {(  0,   0,   0) : 0 ,
             (128,   0,   0) : 1 , #aerplane
             (  0, 128,   0) : 2 , #bicycle
             (128, 128,   0) : 3 , #bird
             (  0,   0, 128) : 4 , #boat
             (128,   0, 128) : 5 , #bottle
             (  0, 128, 128) : 6 , #bus
             (128, 128, 128) : 7 , #car
             ( 64,   0,   0) : 8 , #cat
             (192,   0,   0) : 9 , #chair
             ( 64, 128,   0) : 10, #cow
             (192, 128,   0) : 11, #DiningTable
             ( 64,   0, 128) : 12, #dog
             (192,   0, 128) : 13, #horse
             ( 64, 128, 128) : 14, #motorbike
             (192, 128, 128) : 15, #person
             (  0,  64,   0) : 16, #potted-plant
             (128,  64,   0) : 17, #sheep
             (  0, 192,   0) : 18, #sofa
             (128, 192,   0) : 19, #train
             (  0,  64, 128) : 20 } #monitor

  return palette

def convert_from_color_segmentation(arr_3d):
    arr_3d = cv2.cvtColor(arr_3d,cv2.COLOR_BGR2RGB)
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)
    palette = pascal_palette()
    for c, i in palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i
    return arr_2d



class DatasetVOCAug(gluon.data.Dataset):
    def __init__(self,voc_sdk_root,fortrain, label_scale, len_resize = 256, hw_crop = (256,256)):
        super(DatasetVOCAug,self).__init__()
        self.data_pairs = []
        self.fortrain = fortrain
        self.len_resize = len_resize
        self.hw_crop = hw_crop
        self.label_scale = label_scale
        if fortrain:
            list_file = "train.txt"
        else:
            list_file = "val.txt"
        with open(os.path.join(voc_sdk_root,list_file),'r') as f:
            for line in f:
                line = line.strip()
                if line == "":
                    continue
                #if len(self.data_pairs) >= 200:
                #    break
                image_path = os.path.join(voc_sdk_root,"img/{}.jpg".format(line))
                label_path = os.path.join(voc_sdk_root,"SegmentationClass/{}.png".format(line))
                self.data_pairs.append((image_path,label_path))
        return
    def __len__(self):
        return len(self.data_pairs)


    def __getitem__(self,idx):
        image = cv2.imread(self.data_pairs[idx][0],1)
        label = cv2.imread(self.data_pairs[idx][1],1)
        #print(self.data_pairs[idx][0])
        H,W,C = image.shape
        if H < W:
            h = self.len_resize
            w = h * W / H
        else:
            w = self.len_resize
            h = w * H / W
        w,h = int(w), int(h)
        image = cv2.resize(image,(w,h))
        label = cv2.resize(label,(w,h), interpolation=cv2.INTER_NEAREST)

        h,w = self.hw_crop
        H,W,_ = image.shape
        #print(h,w)
        if not self.fortrain: #no augments
            image = image[0:h,0:w,:]
            label = label[0:h,0:w]
        else:
            if not self.fortrain:
                image = image[0:h,0:w,:]
                label = label[0:h,0:w]
            else:
                #crop
                dx = random.randint(0,W - w)
                dy = random.randint(0,H - h)
                image = image[dy:dy+h, dx:dx+w, :]
                label = label[dy:dy+h, dx:dx+w]

                #rotation
                H,W,_ = image.shape
                deg = random.randint(-20,20)
                H,W,C = image.shape
                center = (H//2, W//2)
                M = cv2.getRotationMatrix2D(center,deg,1.0)
                image = cv2.warpAffine(image,M,(W,H),flags=cv2.INTER_LINEAR)
                label = cv2.warpAffine(label,M,(W,H),flags=cv2.INTER_NEAREST)
                #smooth
                sigma = random.randint(0,100) / 50.0
                if sigma >= 0.5:
                    image = cv2.GaussianBlur(image,(5,5),sigma)


        image = np.float32(image) / 255.0
        image = np.transpose(image,(2,0,1))

        label = cv2.resize(label, (w//self.label_scale, h//self.label_scale), interpolation=cv2.INTER_NEAREST) #encode only
        label = convert_from_color_segmentation(label).astype(np.int64)

        return (image,label)

def get_class_names():
    names = "background,aerplane,bicycle,bird,boat,bottle,bus,car,cat,chair,cow,dining-table,dog,horse,motorbike,person" + \
            ",potted-plant,sheep,sofa,train,monitor"
    return names.split(',')
import platform
def load(batch_size,scale):
    root = os.path.join(os.getenv("ENV_DATASET_DIR"), "VOCdevkit")        
    trainset = DatasetVOCAug(voc_sdk_root=os.path.join(root,"VOCaug/dataset"),fortrain=True,label_scale=scale)
    testset = DatasetVOCAug(voc_sdk_root=os.path.join(root,"VOCaug/dataset"),fortrain=False,label_scale=scale)
    train_iter = gluon.data.DataLoader(trainset,batch_size,shuffle=True,last_batch="rollover",num_workers=3)
    test_iter = gluon.data.DataLoader(testset,batch_size,shuffle=False,last_batch="rollover",num_workers=3)
    return train_iter, test_iter, len(trainset)

if 0:
    train_iter, test_iter, total = load(1)
    print('num of sample: ',total) 
    from collections import defaultdict
    train_dict = defaultdict(int)
    for batch in train_iter:
        image,label = batch
        #image = image[0].asnumpy()
        label = label[0].asnumpy()
        #image = (np.transpose(image,(1,2,0)) * 255).astype(np.uint8)
        #label_uint8 = label.astype(np.uint8) * 10
        #cv2.imshow("image",image)
        #cv2.imshow("label",label_uint8)
        label = list(set(label.flatten().tolist()))
        for l in label:
            train_dict[l] += 1
        #print(label)
        #cv2.waitKey(500)
       # break

    test_dict = defaultdict(int)
    for batch in test_iter:
        image,label = batch
        #image = image[0].asnumpy()
        label = label[0].asnumpy()
        #image = (np.transpose(image,(1,2,0)) * 255).astype(np.uint8)
        #label_uint8 = label.astype(np.uint8) * 10
        #cv2.imshow("image",image)
        #cv2.imshow("label",label_uint8)
        label = list(set(label.flatten().tolist()))
        for l in label:
            test_dict[l] += 1
        #print(label)
        #cv2.waitKey(500)
       # break
       
    sorted_train = sorted(train_dict.items(), key = lambda x: x[0])
    sorted_test = sorted(test_dict.items(), key = lambda x: x[0])
    print(sorted_train)
    print(sorted_test)
        
        

        

if 0:
    train_iter, test_iter, total = load(1)
    print('num of sample: ',total) 
    for batch in train_iter:
        image,label = batch
        image = image[0].asnumpy()
        label = label[0].asnumpy()

        image = (np.transpose(image,(1,2,0)) * 255).astype(np.uint8)
        label = label.astype(np.uint8) * 10
        cv2.imshow("image",image)
        cv2.imshow("label",label)
        cv2.waitKey(-1)
