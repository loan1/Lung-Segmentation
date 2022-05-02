# dataset test: https://github.com/ieee8023/covid-chestxray-dataset

from Script.dataset_custom import LungDataset
from Script.utils import calculate_metrics
from operator import add

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import os
from torch.utils.data import DataLoader

import torch
import numpy as np

#########################################################################################

def Testdataloader():
    mask_path = '../../dataset_lungseg/test/lungVAE-masks/' # 
    img_path = '../../dataset_lungseg/test/images/'
    mask_list = os.listdir(mask_path)
    img_list = os.listdir(img_path)

    # lay ra cac anh co mat na va dua vao list
    img_mask_list = []
    for i in range(len(mask_list)):
        m = mask_list[i].split('_mask.')
        for j in range(len(img_list)):
            img = img_list[j].rsplit('.',1)
            if m[0] == img[0] :
                img_mask_list.append((img_list[j], mask_list[i]))
    
    # ghi file img_mask.txt
    with open('../../dataset_lungseg/test/img_mask.txt','w') as f:
        for i in range(len(img_mask_list)):
            f.write(str(img_mask_list[i])+ '\n') 

    transfms = A.Compose([
        A.Resize(512,512),
        A.Normalize(mean = [0.5],  std = [0.5]),
        ToTensorV2()   
    ])

    test_set = LungDataset(img_mask_list, img_path, mask_path, transfms)

    loader ={
        'test' : DataLoader(
            test_set, 
            batch_size=8,
            shuffle=False
        )
    }   
    return loader

#########################################################################################

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

            pred = pred.reshape(len(pred), 512, 512) # 4, 224, 224
            ynum = ynum.reshape(len(ynum), 512, 512)

            pred = pred > 0.1 #threshold
            pred = np.array(pred, dtype=np.uint8)

            y_true.append(ynum)    
            y_predict.append(pred)

            # chuyển đổi ngược lại với transform
            x = x.cpu().numpy()
            x = x.reshape(len(x), 512, 512)
            x = x*0.5 + 0.5
            x = np.squeeze(x)
            x = np.clip(x, 0, 1)

            image.append(x)

        epoch_jaccard = metrics_score[0]/len(dataloader)
        epoch_acc = metrics_score[1]/len(dataloader)
        epoch_f1 = metrics_score[2]/len(dataloader)
        epoch_recall = metrics_score[3]/len(dataloader)
        epoch_precision = metrics_score[4]/len(dataloader)        

    return image, y_true, y_predict, epoch_jaccard, epoch_f1, epoch_recall, epoch_precision, epoch_acc

#########################################################################################
def save_np(path, image, y_true, y_prect, y_mean ):
    np.save(path + '/images.npy',image)
    np.save(path + '/masks.npy',y_true)
    np.save(path + '/y_predict_5folds.npy',y_prect)
    np.save(path + '/y_predict_mean_5folds.npy',y_mean)
    
#########################################################################################
def mainTest(cp_path, device, path_np, model):
    jaccards, f1s, recalls, precisions, accs = [], [], [], [], []
    y_prect, cp_list = [], []
    cp_fold = ['/UNet0.pt', '/UNet1.pt', '/UNet2.pt', '/UNet3.pt', '/UNet4.pt']

    for i in range(len(cp_fold)):
        cp_list.append(cp_path + cp_fold[i])
   
    modelUNet = model.to(device)

    for fold in range(len(cp_list)):
        checkpoint = torch.load(cp_list[fold])
        modelUNet.load_state_dict(checkpoint)    
             
        ############################################################################################
        image, y_true, y_pred, jaccard, f1, recall, precision, acc = test(Testdataloader()['test'], device, modelUNet, calculate_metrics)

        jaccards.append(jaccard)
        f1s.append(f1)
        recalls.append(recall)
        precisions.append(precision)
        accs.append(acc)   
    
        
        print ('Fold: {} jaccard: {:.4f} - f1: {:.4f} - recall: {:.4f} - precision: {:.4f} - acc: {:.4f}'.format (fold, jaccard, f1, recall, precision, acc))    

        y_prect.append(y_pred)

    y_mean = np.mean(np.stack(y_prect, axis=0), axis=0)
 
    for idx in range(len(y_mean)):
        y_mean[idx] = y_mean[idx] > 0.3  # True False False True 

        y_mean[idx] = y_mean[idx].astype(np.uint8)  # 0 1 1 0

    # imshow(image, y_true, y_prect, y_mean, '../../visualize/testResNet18Gauss/testResNet152.png')
    #####################################################
    save_np(path_np, image, y_true, y_prect, y_mean)

############################################################################################################        

def load_np(path):
    images_np = np.load(path + '/images.npy', allow_pickle = True)
    masks_np = np.load(path + '/masks.npy', allow_pickle = True)
    y_prect = np.load(path +  '/y_predict_5folds.npy', allow_pickle = True)
    y_arg = np.load(path + '/y_predict_mean_5folds.npy', allow_pickle = True)
    return images_np, masks_np, y_prect, y_arg

#############################################################################################################
