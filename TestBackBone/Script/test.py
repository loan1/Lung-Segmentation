from UNetResNet import *
from dataset import LungDataset
from utils import calculate_metrics
from operator import add

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import os
from torch.utils.data import DataLoader
import torch
import numpy as np
import matplotlib.pyplot as plt


def dataloader():

    mask_path = '../dataset_lungseg/test/lungVAE-masks/' #dataset của Phú
    img_path = '../dataset_lungseg/test/images/'
    mask_list = os.listdir(mask_path)
    img_list = os.listdir(img_path)

    img_mask_list = []
    for i in range(len(mask_list)):
        m = mask_list[i].split('_mask.')
        for j in range(len(img_list)):
            img = img_list[j].rsplit('.',1)
            if m[0] == img[0] :
                img_mask_list.append((img_list[j], mask_list[i]))
    
    with open('../dataset_lungseg/test/img_mask.txt','w') as f:
        for i in range(len(img_mask_list)):
            f.write(str(img_mask_list[i])+ '\n') 

    transfms = A.Compose([
        A.Resize(256,256),
        A.Normalize(mean = [0.5],  std = [0.5]),
        ToTensorV2()   
    ])

    test_set = LungDataset(img_mask_list, img_path, mask_path, transfms)

    loader ={
        'test' : DataLoader(
            test_set, 
            batch_size=4,
            shuffle=False
        )
    }   
    # print(len(loader['test']))
    return loader

def test(dataloader, device, model, metric_fn):
   
    with torch.no_grad():
        image, y_true, y_predict = [], [], []
        metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0]
        model.eval()
        for x, y in dataloader:
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)               

            #######################
            #### METRIC###########
            #######################
            y_pred = model(x)
            score = metric_fn(y_pred, y)
            metrics_score = list(map(add, metrics_score, score))  

############################################################################

            pred = y_pred.cpu().numpy() # mask output 
            ynum = y.cpu().numpy()  # mask label

            pred = pred.reshape(len(pred), 256, 256) # 4, 224, 224
            ynum = ynum.reshape(len(ynum), 256, 256)

            pred = pred > 0.3
            pred = np.array(pred, dtype=np.uint8)

            ynum = ynum > 0.3
            ynum = np.array(ynum, dtype=np.uint8)   

            y_true.append(ynum)    
            y_predict.append(pred)

            x = x.cpu().numpy()

            x = x.reshape(len(x), 256, 256)
            x = x*0.5 + 0.5
            x = np.squeeze(x)
            x = np.clip(x, 0, 1)

            image.append(x)


        epoch_jaccard = metrics_score[0]/len(dataloader)
        epoch_f1 = metrics_score[1]/len(dataloader)
        epoch_recall = metrics_score[2]/len(dataloader)
        epoch_precision = metrics_score[3]/len(dataloader)
        epoch_acc = metrics_score[4]/len(dataloader)

    return image, y_true, y_predict, epoch_jaccard, epoch_f1, epoch_recall, epoch_precision, epoch_acc

#########################################################################################

def predict(x, model, device): # dataset 14gb
    model.eval()
    with torch.no_grad():
        x = x.to(device, dtype=torch.float32)
        # y = y.to(device, dtype=torch.float32)

        # output= model(x)
        # _,pred = torch.max(output, 1)

        y_pred = model(x)
        pred = y_pred.cpu().numpy() # mask output 
        # ynum = y.cpu().numpy()  # mask label

        pred = pred.reshape(len(pred), 256, 256)
        # ynum = ynum.reshape(len(ynum), 224, 224)

        pred = pred > 0.5
        pred = np.array(pred, dtype=np.uint8)
  
        x = x.cpu().numpy()
        # print('len x', len(x))
        x = x.reshape(len(x), 256, 256)
        # print(x.shape)
        x = x*0.5 + 0.5
        x = np.squeeze(x)
        # # print(input)
        x = np.clip(x, 0, 1)

    return x, pred
    

def mainTest():
    jaccards, f1s, recalls, precisions, accs = [], [], [], [], []
    y_prect = []
    cp_list = ['../model/UNetResNet34/UNet0.pt', '../model/UNetResNet34/UNet1.pt', '../model/UNetResNet34/UNet2.pt', '../model/UNetResNet34/UNet3.pt', '../model/UNetResNet34/UNet4.pt']

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # model = UNet(in_channels=1, out_channels=1, batch_norm=True)        
    # model = model.to(device)
    
    model = UNet_ResNet.to(device)

    for fold in range(len(cp_list)):
        checkpoint = torch.load(cp_list[fold])
        model.load_state_dict(checkpoint)
         

        ############################################################################################


        image, y_true, y_pred, jaccard, f1, recall, precision, acc = test(dataloader()['test'], device, model, calculate_metrics)

        jaccards.append(jaccard)
        f1s.append(f1)
        recalls.append(recall)
        precisions.append(precision)
        accs.append(acc)        
        
        print ('Fold: {} jaccard: {:.4f} - f1: {:.4f} - recall: {:.4f} - precision: {:.4f} - acc: {:.4f}'.format (fold, jaccard, f1, recall, precision, acc))    

        y_prect.append(y_pred)
    

    y_mean = np.mean(np.stack(y_prect, axis=0), axis=0)
   
    # print(y_mean.shape)
    # print(y_mean[0].shape)

    for idx in range(len(y_mean)):
        y_mean[idx] = y_mean[idx] > 0.7
        y_mean[idx] = y_mean[idx].astype(np.uint8)

    imshow(image, y_true, y_prect, y_mean, '../visualize/testResNet34.png')
    #####################################################

    # print(y_avg[0].shape)
    # print(y_avg)

    # y_avg = y_avg > 0.5
    # y_avg = np.array(y_avg, dtype=np.uint8)
    # print(y_avg)

    # for idx in range(len(y_avg)):
    #     y_avg[idx] = (y_avg[idx] > 0.5) # True False False True 
    #     y_avg[idx] = y_avg[idx].astype(np.uint8) # 0 1 1 0
    
    np.save('../visualize/ResNet34/images.npy',image)
    np.save('../visualize/ResNet34/masks.npy',y_true)
    np.save('../visualize/ResNet34/y_predict_5folds.npy',y_prect)
    np.save('../visualize/ResNet34/y_predict_mean_5folds.npy',y_mean)


    
        ############################################################################################################
        

def imshow(original,true,pred, mean, path):

    plt.figure (figsize = (15, 20))

    for idx in range(4):

        plt.subplot (4,8,idx*8 +1)
        plt.imshow(original[0][idx], cmap='gray')
        # plt.title('Original Image')

        plt.subplot (4,8,idx*8 +2)
        plt.imshow(true[0][idx], cmap='gray')
        # plt.title('True Mask')

        plt.subplot (4,8,idx*8 +3)
        plt.imshow(pred[0][0][idx], cmap='gray')
        # plt.title('Predict Mask')

        plt.subplot (4,8,idx*8 +4)
        plt.imshow(pred[1][0][idx], cmap='gray')
        # plt.title('Predict Mask')

        plt.subplot (4,8,idx*8 +5)
        plt.imshow(pred[2][0][idx], cmap='gray')
        # plt.title('Predict Mask')

        plt.subplot (4,8,idx*8 +6)
        plt.imshow(pred[3][0][idx], cmap='gray')
        # plt.title('Predict Mask')


        plt.subplot (4,8,idx*8 +7)
        plt.imshow(pred[4][0][idx], cmap='gray')
        # plt.title('Predict Mask')

        plt.subplot (4,8,idx*8 +8)
        # print(mean.shape)
        plt.imshow(mean[0][idx], cmap='gray')
        # plt.title('Predict Mean')
        
        plt.savefig(path)
    # plt.title('Original ---- True ---- Predict ---- Predict ---- Predict ---- Predict ---- Predict ---- Mean')
    plt.show()

def mainPredict():
    cp_list = ['../model/original/UNet0.pt', '../model/original/UNet1.pt', '../model/original/UNet2.pt', '../model/original/UNet3.pt', '../model/original/UNet4.pt']
    y = []
    image_tfm = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean = [0.5],  std = [0.5]), 
        ToTensorV2()
    ])

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # model = UNet(in_channels=1, out_channels=1, batch_norm=True)        
    # model = model.to(device)
    model = UNet_ResNet.to(device)
    
    for fold in range(len(cp_list)):
        checkpoint = torch.load(cp_list[fold])
        model.load_state_dict(checkpoint)

        img = Image.open('../dataset_lungseg/predict/1dad3414-88c9-4c56-af5d-3a1488af452c.png').convert('L')
        img = np.array(img, dtype=np.float32)
        img = image_tfm(image = img) 
        img = img['image']
        img = img.unsqueeze(0)
        # print(img.shape)

        x, y_pred = predict(img, model, device)
        y.append(y_pred)
    y_avg = np.mean(y, axis=0)

    # y_avg = y_avg.cpu().detach().numpy()
    y_avg = y_avg > 0.7 # True False False True 
    y_avg = y_avg.astype(np.uint8) # 0 1 1 0
    # y_avg = y_avg.reshape(-1) # flatten
    # print(y_avg.shape)

    # for idx in range(4):
    #     y_avg[idx] = (y_avg[idx] > 0.7) # True False False True 
    #     y_avg[idx] = y_avg[idx].astype(np.uint8) # 0 1 1 0

    plt.figure (figsize = (15, 20))

    plt.subplot (1,7,1)
    plt.imshow(x, cmap='gray')
    plt.title('Original Image')

    plt.subplot (1,7,2)
    plt.imshow(y[0][0], cmap='gray')
    plt.title('Predict Mask1')

    plt.subplot (1,7,3)
    plt.imshow(y[1][0], cmap='gray')
    plt.title('Predict Mask2')

    plt.subplot (1,7,4)
    plt.imshow(y[2][0], cmap='gray')
    plt.title('Predict Mask3')


    plt.subplot (1,7,5)
    plt.imshow(y[3][0], cmap='gray')
    plt.title('Predict Mask4')

    plt.subplot (1,7,6)
    plt.imshow(y[4][0], cmap='gray')
    plt.title('Predict Mask5')

    plt.subplot (1,7,7)
    plt.imshow(y_avg[0], cmap='gray')
    plt.title('Predict Mask')

    plt.show()


if __name__ == '__main__':
    mainTest()
    # images_np = np.load('../visualize/images.npy', allow_pickle=True)
    # masks_np = np.load('../visualize/masks.npy', allow_pickle=True)
    # y_prect = np.load('../visualize/y_predict_5folds.npy', allow_pickle=True)
    # y_arg = np.load('../visualize/y_predict_mean_5folds.npy', allow_pickle=True)
    
    # y_mean = np.mean(np.stack(y_prect, axis=0), axis=0)
   
    # # print(y_mean.shape)
    # # print(y_mean[0].shape)

    # for idx in range(len(y_mean)):
    #     y_mean[idx] = y_mean[idx] > 0.7
    #     y_mean[idx] = y_mean[idx].astype(np.uint8)

    # imshow(images_np, masks_np, y_prect, y_mean, '../visualize/test07.png')


    # y_mean =
    #     y_avg = y_avg > 0.7 # True False False True 
    # y_avg = y_avg.astype(np.uint8) # 0 1 1 0
