from torch.utils.data import Dataset
#from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing import image
import numpy as np
#from utils import np2tensor
import random
from torchvision import transforms
from torch.utils.data import DataLoader
import os

class FIW2(Dataset):
    def __init__(self,
                 sample_path,
                 transform=None,mode='train'):
        self.mode = mode
        self.sample_path=sample_path
        self.transform=transform
        self.sample_list=self.load_sample()
        self.bias=0

    def load_sample(self):
        sample_list= []
        f = open(self.sample_path, "r+", encoding='utf-8')
        while True:
            line = f.readline().replace('\n', '')
            if not line:
                break
            else:
                # pdb.set_trace()
                if self.mode=='train':
                    tmp = line.split(' ')
                    sample_list.append([tmp[1], tmp[2], tmp[3], tmp[4], tmp[5]])
                elif self.mode == 'val':
                    tmp = line.split(' ')
                    sample_list.append([tmp[1], tmp[2], tmp[3], tmp[4]])

        f.close()
        return sample_list

    def __len__(self):
        return len(self.sample_list)

    def read_image(self, path):
        abs_path = os.path.join('/home/yyh/zyy/FIW',path)
        img = image.load_img(abs_path, target_size=(112, 112))
        #img = tf.image.resize(path, [112, 112])
        return img

    def set_bias(self,bias):
        self.bias=bias

    def preprocess(self, img):
        return np.transpose(img, (2, 0, 1))

    def __getitem__(self, item):
        if self.mode == 'train':
            sample = self.sample_list[item+self.bias]
            img1,img2=self.read_image(sample[0]),self.read_image(sample[1])        
            if self.transform is not None:
                img1,img2 = self.transform(img1),self.transform(img2)
            img1, img2 = self.preprocess(np.array(img1, dtype=float)), \
                        self.preprocess(np.array(img2, dtype=float))
            img1 = ((img1/255.0) - 0.5)/0.5
            img2 = ((img1/255.0) - 0.5)/0.5
            # pdb.set_trace()
            kin_label =np.array(sample[4], dtype=float)
            # pdb.set_trace()
            # kin_label = np2tensor(np.array(sample[3], dtype=float))
            label = np.array(sample[3],dtype=float)
            # quality = sample[5]
            class_ = sample[2]
        
            return img1, img2, label, kin_label,class_
        else:
            sample = self.sample_list[item+self.bias]
            img1,img2=self.read_image(sample[0]),self.read_image(sample[1])        
            if self.transform is not None:
                img1,img2 = self.transform(img1),self.transform(img2)
            img1, img2 = self.preprocess(np.array(img1, dtype=float)), \
                        self.preprocess(np.array(img2, dtype=float))
            img1 = ((img1/255.0) - 0.5)/0.5
            img2 = ((img1/255.0) - 0.5)/0.5
            # pdb.set_trace()
            # pdb.set_trace()
            # kin_label = np2tensor(np.array(sample[3], dtype=float))
            label = np.array(sample[3],dtype=float)
            # quality = sample[5]
            class_ = sample[2]
        
            return img1, img2, label,class_
