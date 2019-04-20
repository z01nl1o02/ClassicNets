from mxnet import gluon
import os,cv2,pdb
import numpy as np
import random
import xml.etree.ElementTree as ET


class DETECT_VOC(gluon.data.Dataset):
    def __init__(self, image_set, year, resize = (256,256), devkit_path='E:\\dataset\\VOCdevkit\\'):
        super(DETECT_VOC,self).__init__()
        self._year = year
        self._resize = resize
        self._image_set = image_set
        self._objects_per_image = 45
        self._devkit_path = devkit_path
        self._data_path = os.path.join(self._devkit_path, 'VOC' + self._year)
        self._classes = (
                         'aeroplane', 'bicycle', 'bird', 'boat',
                         'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse',
                         'motorbike', 'person', 'pottedplant',
                         'sheep', 'sofa', 'train', 'tvmonitor')

        if image_set == "trainval":
            self.fortrain = True
        else:
            self.fortrain = False
        self._class_to_ind = dict(zip(self._classes, range(len(self._classes))))
        self._image_ext = '.jpg'
        self._image_index = self._load_image_set_index()
        
        assert os.path.exists(self._devkit_path), \
                'VOCdevkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)
                                        
    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, 'JPEGImages',index + self._image_ext)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
        image_set_file = os.path.join(self._data_path, 'ImageSets', 'Main',
                                      self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index
        
    def _load_pascal_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        filename = os.path.join(self._data_path,'Annotations',index+'.xml')
        tree = ET.parse(filename)
        size = tree.find("size")
        width = float(size.find("width").text)
        height = float(size.find("height").text)
        objs = tree.findall('object')
        #if not self.config['use_diff']:
            # Exclude the samples labeled as difficult
        #    non_diff_objs = [
            #    obj for obj in objs if int(obj.find('difficult').text) == 0]
            # if len(non_diff_objs) != len(objs):
            #     print 'Removed {} difficult objects'.format(
            #         len(objs) - len(non_diff_objs))
        #    objs = non_diff_objs
        #num_objs = len(objs)
        
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
        return len(self._image_index)
        
    def __getitem__(self,idx):
        image_name = self._image_index[idx]
        targets = self._load_pascal_annotation(image_name)
        path = self.image_path_from_index(image_name)
        img = cv2.imread(path,1)
        
        
        if self.fortrain:
            interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, \
                              cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
        else:
            interp_methods = [cv2.INTER_LINEAR]
        interp_method = interp_methods[int(np.random.uniform(0, 1) * len(interp_methods))]              
        img = cv2.resize(img,self._resize,interp_method)
        #pdb.set_trace()
        if self.fortrain:
            if np.random.uniform(0, 1) > 0.5:
                img = cv2.flip(img, 1)
                tmp = 1.0 - targets[:, 1]
                targets[:, 1] = 1.0 - targets[:, 3]
                targets[:, 3] = tmp
        
        img = np.transpose(img,(2,0,1))
        img = np.float32(img) / 255
        return img,targets
        
        
        
        
def load(year,batch_size):
    trainset = DETECT_VOC("trainval",year)
    testset = DETECT_VOC("test",year)
    train_iter = gluon.data.DataLoader(trainset,batch_size,shuffle=True,last_batch="rollover")
    test_iter = gluon.data.DataLoader(testset,batch_size,shuffle=False,last_batch="rollover")
    return train_iter, test_iter, len(trainset._classes)
   
if 0:
    dataloader, _, _ = load("2007",1)
    #print("---")
    for data in dataloader:
        img,targets = data
        img,targets = img.asnumpy()[0], targets.asnumpy()[0]
        img = np.uint8(img * 255)
        img = np.transpose(img, (1,2,0))
        H,W,C = img.shape
        #print(img.shape)
        for target in targets:
            cls,x0,y0,x1,y1 = (target * np.array([1,W,H,W,H])).astype(np.int32)
            if cls < 0:
                continue 
            img = cv2.rectangle(img,(x0,y0),(x1,y1),(255,255,0), 2)
        cv2.imshow("vis",img)
        cv2.waitKey(500)
    
    

class DETECT_VOC_FORWARD:
    def __init__(self, image_set, year, resize = (256,256), devkit_path='E:\\dataset\\VOCdevkit\\'):
        super(DETECT_VOC_FORWARD,self).__init__()
        self._year = year
        self._resize = resize
        self._image_set = image_set
        self._devkit_path = devkit_path
        self._objects_per_image = 64
        self._data_path = os.path.join(self._devkit_path, 'VOC' + self._year)
        self._classes = (
                         'aeroplane', 'bicycle', 'bird', 'boat',
                         'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse',
                         'motorbike', 'person', 'pottedplant',
                         'sheep', 'sofa', 'train', 'tvmonitor')

        
        self._class_to_ind = dict(zip(self._classes, range(len(self._classes))))
        self._image_ext = '.jpg'
        self._image_index = self._load_image_set_index()
        
        assert os.path.exists(self._devkit_path), \
                'VOCdevkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)
    def classes(self):
        return self._classes 
        
    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, 'JPEGImages',index + self._image_ext)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
        image_set_file = os.path.join(self._data_path, 'ImageSets', 'Main',
                                      self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index
        
    def _load_pascal_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        filename = os.path.join(self._data_path,'Annotations',index+'.xml')
        tree = ET.parse(filename)
        size = tree.find("size")
        width = float(size.find("width").text)
        height = float(size.find("height").text)
        objs = tree.findall('object')
        #if not self.config['use_diff']:
            # Exclude the samples labeled as difficult
        #    non_diff_objs = [
            #    obj for obj in objs if int(obj.find('difficult').text) == 0]
            # if len(non_diff_objs) != len(objs):
            #     print 'Removed {} difficult objects'.format(
            #         len(objs) - len(non_diff_objs))
        #    objs = non_diff_objs
        #num_objs = len(objs)
        
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
        return len(self._image_index)
        
    def __getitem__(self,idx):
        image_name = self._image_index[idx]
        targets = self._load_pascal_annotation(image_name)
        path = self.image_path_from_index(image_name)
        src = cv2.imread(path,1)
        img = cv2.resize(src,self._resize)
        img = np.transpose(img,(2,0,1))
        img = np.float32(img) / 255
        return img,targets, src, image_name + self._image_ext #feature, target, origin_image, imagename
    
