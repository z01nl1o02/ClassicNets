from mxnet import gluon
import os,sys
import cv2
import numpy as np
import random
import xml.etree.ElementTree as ET
import math

class DETECT_VOC(gluon.data.Dataset):
    def __init__(self, image_set, years, fortrain,resize = (300,300), devkit_path='/home/c001/data/VOCdevkit/'):
        super(DETECT_VOC,self).__init__()
        self._resize = resize
        self._objects_per_image = 45
        self._classes = (
                         'aeroplane', 'bicycle', 'bird', 'boat',
                         'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse',
                         'motorbike', 'person', 'pottedplant',
                         'sheep', 'sofa', 'train', 'tvmonitor')

        self._fortrain = fortrain
        self._use_difficult = False

        self._class_to_ind = dict(zip(self._classes, range(len(self._classes))))
        self._image_ext = '.jpg'
        self._paths = []
        for year in years.split('_'): #2007_2012
            data_path = os.path.join(devkit_path, 'VOC' + year)
            assert os.path.exists(data_path), 'dataset path does not exist: {}'.format(data_path)
            self._paths.extend( self._load_paths(data_path, image_set )   )


    def _load_paths(self,data_path,image_set):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
        image_set_file = os.path.join(data_path, 'ImageSets', 'Main', image_set + '.txt')
        if not os.path.exists(image_set_file):
            print('Path does not exist: {}'.format(image_set_file))
            return []

        image_dir = os.path.join(data_path, "JPEGImages")
        xml_dir = os.path.join(data_path,"Annotations")
        paths = []
        with open(image_set_file) as f:
            for line in f:
                index = line.strip()
                paths.append( ( os.path.join(image_dir, index + self._image_ext),  os.path.join( xml_dir, index + ".xml"  ) ) )
        return paths
        
    def _load_pascal_annotation(self, xml_path):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        tree = ET.parse(xml_path)
        size = tree.find("size")
        width = float(size.find("width").text)
        height = float(size.find("height").text)
        objs = tree.findall('object')
        if not self._use_difficult:
            # Exclude the samples labeled as difficult
            non_diff_objs = [obj for obj in objs if int(obj.find('difficult').text) == 0]
            #if len(non_diff_objs) != len(objs):
            #     print('Removed {} difficult objects'.format(len(objs) - len(non_diff_objs)))
            objs = non_diff_objs
        num_objs = len(objs)
        
        targets = np.zeros( (self._objects_per_image, 5), dtype=np.float32 ) - 1 #default 
        
        for ix, obj in enumerate(objs):
            if ix >= self._objects_per_image:
                print("exceeding maximum objects number %d > %d"%(len(objs),self._objects_per_image))
                break
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            #print(self._class_to_ind)
            cls = self._class_to_ind[obj.find('name').text.lower().strip()]
            targets[ix,:] = np.asarray([cls,x1/width,y1/height,x2/width,y2/height])
        return targets
    def __len__(self):
        return len(self._paths)
     
    def get_name_at(self,idx):
        image_path, xml_path = self._paths[idx]
        return os.path.split(image_path)[-1]

    def get_origin_image_at(self,idx):
        image_path, xml_path = self._paths[idx]
        img = cv2.imread(image_path,1)
        return img
    #https://mxnet.incubator.apache.org/api/python/image/image.html
    #https://mxnet.incubator.apache.org/versions/master/tutorials/python/types_of_data_augmentation.html?highlight=contrastjitteraug
    def BrightnessJitterAug(self, img, brightness = 0.5):
        img = img.astype(np.float32)
        alpha = 1.0 + random.uniform(-brightness, brightness)
        return np.clip(img * alpha,0,255)
    def ContrastJitterAug(self,img,contrast = 0.5):
        coef = np.asarray(  [[[0.299,0.587,0.114]]]    )
        alpha = 1.0 + random.uniform(-contrast, contrast)
        gray = img * coef
        gray = (3.0*(1.0-alpha) / gray.size) * np.sum(gray)
        img *= alpha
        img += gray
        return np.clip(img,0,255)
    def SaturationJitterAug(self,src,saturation=0.5):
        coef = np.asarray(  [[[0.299,0.587,0.114]]]    )
        alpha = 1.0 + random.uniform(-saturation, saturation)
        gray = src * coef
        gray = np.sum(gray, axis=2,keepdims=True)
        gray *= (1.0 - alpha)
        src *= alpha
        src += gray
        return np.clip(src,0,255)
    def HueJitterAug(self,src, hue = 0.5):
        tyiq = np.array([[0.299, 0.587, 0.114],
                              [0.596, -0.274, -0.321],
                              [0.211, -0.523, 0.311]])
        ityiq = np.array([[1.0, 0.956, 0.621],
                               [1.0, -0.272, -0.647],
                               [1.0, -1.107, 1.705]])

        alpha = random.uniform(-hue, hue)
        u = np.cos(alpha * np.pi)
        w = np.sin(alpha * np.pi)
        bt = np.array([[1.0, 0.0, 0.0],
                       [0.0, u, -w],
                       [0.0, w, u]])
        t = np.dot(np.dot(ityiq, bt), tyiq).T
        src = np.dot(src, np.array(t))
        return np.clip(src,0,255)

    def RandomGrayAug(self,src):
        mat = np.array([[0.21, 0.21, 0.21],
                             [0.72, 0.72, 0.72],
                             [0.07, 0.07, 0.07]])
        src = np.dot(src,mat)
        return np.clip(src,0,255)
    def SmoothAug(self,src):
        sigmaX, sigmaY = random.random() + 0.5, random.random() + 0.5
        src = cv2.GaussianBlur(src,(5,5),sigmaX,sigmaY).astype(np.float32)
        return src
    def BatchSample(self,src,bbox):
        H,W,C = src.shape
        min_ious = [0.1,0.3,0.5,0.7,0.9]
        r = random.randint(0, len(min_ious) - 1)
        iou_thresh = min_ious[r]
        valid_bbox_indices = (bbox[:,0] >= 0)
        for _ in range(50):
            scale = random.uniform(0.3,1.)
            min_ratio,max_ratio = max(0.5, scale*scale), min(2, 1./scale/scale)
            ratio = math.sqrt(random.uniform(min_ratio,max_ratio))
            newW,newH = int(scale * ratio * W), int(scale / ratio * H)
            left,top = random.randint(0,W - newW), random.randint(0, H-newH)
            roi = np.array( [left,top, left + newW, top + newH]   ) / np.array([W,H,W,H])
            ious = []
            for (cls,x0,y0,x1,y1) in bbox[valid_bbox_indices]:
                l,t = max(x0,roi[0]),max(y0,roi[1])
                r,b = min(x1,roi[2]), min(y1,roi[3])
                w,h = r - l, b - t
                if w < 0 or h < 0:
                    ious.append(-1)
                else:
                    ious.append( (newW * newH * 1.0 / W / H + (x1-x0) * (y1-y0)  -  w * h) * 1.0 / (x1-x0 + 1) / (y1-y0+1) )
            if np.max(ious) <  iou_thresh:
                continue
            centers = (bbox[:,1:3] + bbox[:,3:]) / 2.
            indices = np.logical_and(centers < roi[2:], centers > roi[:2]).all(axis=1)
            if len(bbox[indices]) < 1:
                continue
            indices = np.logical_not( np.logical_and(valid_bbox_indices, indices) )

            #vis = src.copy().astype(np.uint8)
            #for (cls, x0,y0,x1,y1) in bbox:
            #    if cls < 0:
            #        continue
            #    x0,y0,x1,y1 = (np.asarray([x0,y0,x1,y1]) * np.asarray([W,H,W,H])).astype(np.int32)
            #    cv2.rectangle(vis,(x0,y0),(x1,y1),(255,0,0),3)

            bbox = bbox * np.array([1,W,H,W,H])
            roi *= np.array([W,H,W,H])
            roi = roi.astype(np.int32)
            #print(bbox[0], roi)
            bbox[:,1:3] = np.maximum(bbox[:,1:3], roi[:2])
            bbox[:,1:3] -= roi[:2]
            bbox[:,3:] = np.minimum(bbox[:,3:], roi[2:])
            bbox[:,3:] -= roi[:2]
            #print(bbox[0], roi)
            bbox[indices] = -1



            #cv2.rectangle(vis,(roi[0],roi[1]), (roi[2],roi[3]), (0,0,255),2)
            #cv2.imshow("src",vis)

            src = src[roi[1]:roi[3], roi[0]:roi[2],:]


            H,W,C = src.shape
            bbox = (bbox / np.array([1,W,H,W,H])).astype(np.float32)
           # vis = src.copy().astype(np.uint8)
           # for (cls, x0,y0,x1,y1) in bbox:
           #     if cls < 0:
           #         continue
           #     x0,y0,x1,y1 = (np.asarray([x0,y0,x1,y1]) * np.asarray([W,H,W,H])).astype(np.int32)
           #     print(x0,y0,x1,y1)
           #     cv2.rectangle(vis,(x0,y0),(x1,y1),(255,0,0),3)
           # cv2.imshow("src-2",vis)
            break
        return src,bbox



    def ExpandImage(self,src, bbox):
        H,W,C = src.shape
        for _ in range(50):
            scale = random.uniform(1,4)
            min_ratio = max(0.5, 1./scale/scale)
            max_ratio = min(2, scale * scale)
            ratio = math.sqrt(random.uniform(min_ratio, max_ratio))
            ws,hs = scale * ratio, scale/ratio
            if ws < 1 or hs < 1:
                continue
            newW,newH = int(ws * W), int(hs * H)

            left, top = random.randint(0, newW - W) , random.randint(0, newH - H)
            bbox[:,1:]  = bbox[:,1:] * np.array([W,H,W,H])
            bbox[:,1:3] = bbox[:,1:3] + np.array([left,top])
            bbox[:,3:] = bbox[:,3:] + np.array([left,top])

            expand_image = np.zeros((newH,newW,C), dtype=np.float32)
            expand_image[:,:,0] =  src[:,:,0].mean()
            expand_image[:,:,1] =  src[:,:,1].mean()
            expand_image[:,:,2] =  src[:,:,2].mean()
            expand_image[top:top+H, left:left+W,:] = src

            src = expand_image
            bbox[:,1:]  = bbox[:,1:] / np.array([newW,newH,newW,newH])
            mask = (bbox[:,0] < 0)
            bbox[mask] = -1
            bbox = bbox.astype(np.float32)
            break
        return src,bbox

    def data_aug_shuffle(self,src,bbox,p = 0.5):
        src = src.astype(np.float32)
        if random.random() < p:
            src = self.BrightnessJitterAug(src)
        if random.random() < p:
            src = self.ContrastJitterAug(src)
        if random.random() < p:
            src = self.SaturationJitterAug(src)
        if random.random() < p:
            src = self.HueJitterAug(src)
        if random.random() < p:
            src = self.SmoothAug(src)
        if random.random() < p:
            src = self.RandomGrayAug(src)
        if random.random() < p:
            src,bbox = self.ExpandImage(src,bbox)
        if random.random() < p:
            src,bbox = self.BatchSample(src,bbox)
        return src,bbox

    def __getitem__(self,idx):
        image_path, xml_path = self._paths[idx]
        targets = self._load_pascal_annotation(xml_path)
        img = cv2.imread(image_path,1)

        if self._fortrain:
            if np.random.uniform(0, 1) > 0.5: #flip
                img = cv2.flip(img, 1)
                tmp = 1.0 - targets[:, 1]
                targets[:, 1] = 1.0 - targets[:, 3]
                targets[:, 3] = tmp
            img,targets = self.data_aug_shuffle(img,targets,0.5)

        img = img.astype(np.uint8)
        if self._fortrain:
            interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, \
                              cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
        else:
            interp_methods = [cv2.INTER_LINEAR]
        interp_method = interp_methods[int(np.random.uniform(0, 1) * len(interp_methods))]
        img = cv2.resize(img, self._resize, interp_method)

        img = np.transpose(img,(2,0,1))
        img = np.float32(img) / 255
        #print(img.dtype, targets.dtype)
        return img,targets
        
        
        
        
def load(years,batch_size):
    trainset = DETECT_VOC("trainval",years,True)
    testset = DETECT_VOC("test",years,False)
    print("train: ",len(trainset))
    train_iter = gluon.data.DataLoader(trainset,batch_size,shuffle=True,last_batch="rollover",num_workers=-1)
    test_iter = gluon.data.DataLoader(testset,batch_size,shuffle=False,last_batch="rollover",num_workers=-1)
    return train_iter, test_iter, trainset._classes



if 0:
    dataloader, _, _ = load("2007_2012",1)
    for data in dataloader:
        img,targets = data
        img,targets = img.asnumpy()[0], targets.asnumpy()[0]
        img = np.uint8(img * 255)
        img = np.transpose(img, (1,2,0))
        img = img.copy()
        H,W,C = img.shape
        for target in targets:
            cls,x0,y0,x1,y1 = (target * np.array([1,W,H,W,H])).astype(np.int32)
            if cls < 0:
                continue 
            cv2.rectangle(img,(x0,y0),(x1,y1),(255,255,0), 2)
        cv2.imshow("vis",img)
        cv2.waitKey(-2000)
    
    

    
