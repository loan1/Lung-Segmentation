
#  https://drive.google.com/file/d/1ffbbyoPf-I3Y0iGbBahXpWqYdGd7xxQQ/view
from PIL import Image 
from torch.utils.data import Dataset
import numpy as np
import os

# https://minhng.info/tutorials/xu-ly-anh-bo-loc-lam-mo-blur.html
# https://viblo.asia/p/xu-li-anh-thuat-toan-can-bang-histogram-anh-GrLZDOogKk0
# https://colab.research.google.com/drive/18OTZTjFGzEK3_x3WJlPt0Guu3pyvhgCQ#scrollTo=qJAGI7b_p2Qh

import cv2

def Gauss_His(img):
    # print('type: ',type(img))
    img = cv2.GaussianBlur(img, (3,3), cv2.BORDER_DEFAULT)
    # print('img shap: ', img_blur.shape)
    # grayimg = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
    # img_hist = cv2.equalizeHist(grayimg)
    # img = cv2.equalizeHist(img)

    return img

class LungDataset(Dataset):
    def __init__(self, img_mask_list, img_folder, mask_folder, transform = None): # 'Initialization'
        self.img_mask_list = img_mask_list
        self.img_folder = img_folder
        self.mask_folder = mask_folder
        self.transform = transform
  
    def __len__(self):  # 'Denotes the total number of samples'
        return len(self.img_mask_list)

    def __getitem__(self,index): # 'Generates one sample of data'      

        images_names, masks_names = self.img_mask_list[index]
        images = Image.open(self.img_folder +  images_names).convert('L')# grey # kiểu PIL images
        masks = Image.open(self.mask_folder + masks_names).convert('L') # binary kiểu PIL images

        images = np.array(images, dtype=np.float32) # đổi qua numpy array kiểu float 32
        masks = np.array(masks, dtype=np.float32) # 
        masks[masks == 255] = 1 # nếu giá trị pixcel == 255 thì đưa về 1
        
        if self.transform != None:
            aug = self.transform(image = images, mask = masks)
            images = aug['image']
            masks = aug['mask']
        # masks = masks.long()
        # masks = torch.unsqueeze(masks,0)
        # print(masks.shape)
        return images, masks # chua 1 cap

class DatasetPredict(Dataset):
    def __init__(self, img_folder,  transform = None): # 'Initialization'

        self.img_folder = img_folder
        self.transform = transform
  
    def __len__(self):  # 'Denotes the total number of samples'
        return len(os.listdir(self.img_folder))

    def __getitem__(self,index): # 'Generates one sample of data'      

        images_names = os.listdir(self.img_folder)
        # images = Image.open(self.img_folder +  images_names[index]).convert('L')# grey # kiểu PIL images
        images = cv2.imread(self.img_folder +  images_names[index], 0) # #############

        # images = np.array(images, dtype=np.float32) # đổi qua numpy array kiểu float 32
        # print('image shape: ', images.shape)
        images =  Gauss_His(images)    
        # print('img shape 2: ', images.shape)
        if self.transform != None:
            aug = self.transform(image = images)            
            images = aug['image']

        # print('images: ', images.shape) # torch.Size([1, 256, 256])
        return images # chua 1 anh

        
