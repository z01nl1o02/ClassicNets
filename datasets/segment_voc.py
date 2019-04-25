from mxnet import gluon
import os,cv2
import numpy as np
import random

def pascal_palette_all_classes(): #RGB mode
  palette = {(  0,   0,   0) : 0 ,
             (  1,   1,   1) : 1 , #hat
             (  2,   2,   2) : 2 , #hair           
             (  3,   3,   3) : 3 , #sunglass       
             (  4,   4,   4) : 4 , #upper-clothes
             (  5,   5,   5) : 5 , #skirt
             (  6,   6,   6) : 6 , #pants
             (  7,   7,   7) : 7 , #dress 
             (  8,   8,   8) : 8 , #belt
             (  9,   9,   9) : 9 , #left-shoe
             (  10,   10,   10) : 10, #right-shoe 
             (  11,   11,   11) : 11, #face 
             (  12,   12,   12) : 12, #left-leg 
             (  13,   13,   13) : 13, #right-leg
             (  14,   14,   14) : 14, #left-arm 
             (  15,   15,   15) : 15, #right-arm
             (  16,   16,   16) : 16, #bag
             (  17,   17,   17) : 17, #scarf 
             }

  return palette
  
def pascal_palette_two_classes(): #RGB mode
  palette = {(  0,   0,   0) : 0 ,
             (  1,   1,   1) : 1 , #hat
             (  2,   2,   2) : 1 , #hair           
             (  3,   3,   3) : 1 , #sunglass       
             (  4,   4,   4) : 1 , #upper-clothes
             (  5,   5,   5) : 1 , #skirt
             (  6,   6,   6) : 1 , #pants
             (  7,   7,   7) : 1 , #dress 
             (  8,   8,   8) : 1 , #belt
             (  9,   9,   9) : 1 , #left-shoe
             (  10,   10,   10) : 1, #right-shoe 
             (  11,   11,   11) : 1, #face 
             (  12,   12,   12) : 1, #left-leg 
             (  13,   13,   13) : 1, #right-leg
             (  14,   14,   14) : 1, #left-arm 
             (  15,   15,   15) : 1, #right-arm
             (  16,   16,   16) : 1, #bag
             (  17,   17,   17) : 1, #scarf 
             }

  return palette  

  
def convert_from_color_segmentation(arr_3d,number_classes):
    arr_3d = cv2.cvtColor(arr_3d,cv2.COLOR_BGR2RGB)
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)
    if number_classes == 2:
        palette = pascal_palette_two_classes()
    else:
        palette = pascal_palette()
    for c, i in palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i
    return arr_2d



class DatasetVOC(gluon.data.Dataset):
    def __init__(self,voc_sdk_root,fortrain, label_scale, len_resize = 128, hw_crop = (128,128)):
        super(DatasetVOC,self).__init__()
        self.data_pairs = []
        self.fortrain = fortrain
        self.len_resize = len_resize
        self.hw_crop = hw_crop
        self.label_scale = label_scale
        self.number_classes = 2
        if fortrain:
            list_file = "ImageSets/Segmentation/train.txt"
        else:
            list_file = "ImageSets/Segmentation/val.txt"
        with open(os.path.join(voc_sdk_root,list_file),'r') as f:
            for line in f:
                line = line.strip()
                if line == "":
                    continue
                #if len(self.data_pairs) >= 200:
                #    break
                image_path = os.path.join(voc_sdk_root,"JPEGImages/{}.jpg".format(line))
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
        label = convert_from_color_segmentation(label,self.number_classes).astype(np.int64)

        return (image,label)

def get_class_names():
    names = "background,hat,hair,sunglass,upper-clothes,skirt,pants,dress,belt,left-shoe,right-shoe,face,left-leg,right-leg,left-arm,right-arm" + \
            ",bag,scarf"
    return names.split(',')
import platform
def load(batch_size,scale):
    root = "G:/seg/"
    if platform.system() == "Linux":
        root = "data/VOCdevkit/"
    trainset = DatasetVOC(voc_sdk_root=os.path.join(root,"humanparsing"),fortrain=True,label_scale=scale)
    testset = DatasetVOC(voc_sdk_root=os.path.join(root,"humanparsing"),fortrain=False,label_scale=scale)
    train_iter = gluon.data.DataLoader(trainset,batch_size,shuffle=True,last_batch="rollover")
    test_iter = gluon.data.DataLoader(testset,batch_size,shuffle=False,last_batch="rollover")
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
