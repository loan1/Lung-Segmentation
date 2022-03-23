from models import *
from dataset import DatasetPredict, LungDataset
from utils import calculate_metrics
from operator import add

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import os
from torch.utils.data import DataLoader

import torch
import numpy as np
import matplotlib.pyplot as plt

def dataloaderPre(): # data 14gb

    data_dir = '../../dataset_lungseg/predict/'

    transfms = A.Compose([
        A.Resize(256,256),
        A.Normalize(mean = [0.5],  std = [0.5]),
        ToTensorV2()   
    ])

    dataNegative = DatasetPredict(data_dir + 'Negative50/', transform = transfms)
    dataPositive = DatasetPredict(data_dir + 'Positive50/', transform = transfms)

    loader ={
        'Negative' : DataLoader(
            dataNegative, 
            batch_size=4,
            shuffle=False
        ),
        'Positive' : DataLoader(
            dataPositive, 
            batch_size=4,
            shuffle=False
        )
    }   
    # print(len(loader['test']))
    return loader



def predict(dataloader, model, device): # dataset 14gb
    model.eval()
    with torch.no_grad():
        original, image, y_predict = [], [], []
        for x in dataloader:
            # x = x.to(device, dtype=torch.float32)
            x = x.to(device)
            # x = x.cpu().numpy()        
    
            # print(x.shape) # torch.Size([4, 1, 256, 256])
            y_pred = model(x)
            pred = y_pred.cpu().numpy() # mask output 
            # ynum = y.cpu().numpy()  # mask label
            # print('pre1: ',pred.shape) # pre1:  (4, 1, 256, 256)
            

            pred = pred.reshape(len(pred), 256, 256)

            # ynum = ynum.reshape(len(ynum), 224, 224)
            # print('pre2: ', pred.shape) # pre2:  (4, 256, 256)
            pred = pred > 0.3
            pred = np.array(pred, dtype=np.uint8)
            y_predict.append(pred)

            x = x.cpu().numpy()
            # print('len x', len(x))
            x = x.reshape(len(x), 256, 256)
            # print(x.shape)
            x = x*0.5 + 0.5
            x = np.squeeze(x)
            # # print(input)
            x = np.clip(x, 0, 1)
            image.append(x)

            

    return image, y_predict

def mainPredict():
    cp_list = ['../../model/UNetResNet18/UNet0.pt', '../../model/UNetResNet18/UNet1.pt', '../../model/UNetResNet18/UNet2.pt', '../../model/UNetResNet18/UNet3.pt', '../../model/UNetResNet18/UNet4.pt']
    
    y = []
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    modelUNet = UNet_ResNet18.to(device)
    
    for fold in range(len(cp_list)):
        checkpoint = torch.load(cp_list[fold])
        modelUNet.load_state_dict(checkpoint)

        # img = Image.open('../dataset_lungseg/predict/1dad3414-88c9-4c56-af5d-3a1488af452c.png').convert('L')
        # img = np.array(img, dtype=np.float32)
        # img = image_tfm(image = img) 
        # img = img['image']
        # img = img.unsqueeze(0)

        x, y_pred = predict(dataloaderPre()['Positive'], modelUNet, device)
        
        # print('x: ', x.shape)
        # print('y_pred: ', y_pred.shape)
        y.append(y_pred)

    y_mean = np.mean(np.stack(y, axis=0), axis=0) 

    for idx in range(len(y_mean)):
        y_mean[idx] = y_mean[idx] > 0.3
        y_mean[idx] = y_mean[idx].astype(np.uint8)

    # print('x.shape ',x.shape)
    # # print('y[1].shape ',y[1].shape)
    # print('y_mean.shape ',y_mean.shape)


    imshowPre(x, y, y_mean, '../../visualize/InferResNet18Gauss/InferResNet18.png')

def imshowPre(x ,pred, mean, path):

    for i in range(13):

        plt.figure (figsize = (15, 20))

        for idx in range(4):

            plt.subplot (4,7,idx*7 +1)
            # print(original.shape)
            plt.imshow(x[i][idx], cmap='gray') 

            plt.subplot (4,7,idx*7 +2)
            plt.imshow(pred[0][i][idx], cmap='gray')
            # plt.title('Predict Mask')

            plt.subplot (4,7,idx*7 +3)
            plt.imshow(pred[1][i][idx], cmap='gray')
            # plt.title('Predict Mask')

            plt.subplot (4,7,idx*7 +4)
            plt.imshow(pred[2][i][idx], cmap='gray')
            # plt.title('Predict Mask')

            plt.subplot (4,7,idx*7 +5)
            plt.imshow(pred[3][i][idx], cmap='gray')
            # plt.title('Predict Mask')


            plt.subplot (4,7,idx*7 +6)
            plt.imshow(pred[4][i][idx], cmap='gray')
            # plt.title('Predict Mask')

            plt.subplot (4,7,idx*7 +7)
            # print(mean.shape)
            plt.imshow(mean[i][idx], cmap='gray')
            # plt.title('Predict Mean')
            
        plt.savefig(path.replace('.png', '_' +str(i) + '.png'))
        plt.show()

if __name__ == '__main__':

    mainPredict()
    ############################################################################
    # data_dir = '../../dataset_lungseg/predict/'
    # img_folder = data_dir + 'Negative50/'

    # # print(len(os.listdir(img_folder))) # = 50

    # images_names = os.listdir(img_folder) # = list 50 tên ảnh
    # # print('images_names: ', images_names)
    # images = Image.open(img_folder +  images_names[20]).convert('L')
    # # print('images: ', images)
    # images = np.array(images, dtype=np.float32)
    # print('images: ', images.shape)
    # img = next(iter(dataloaderPre()['Positive']))
    # # print(img)
    # for i in range(len(img)):
    #     plt.imshow(img[i][0], cmap='gray')
    #     plt.show()