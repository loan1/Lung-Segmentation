from script.utils import *
from script.dataset import *
from script.visualize import *

#importing the libraries
import os
import numpy as np
import torchvision.transforms as transforms
#for reading and displaying images
import matplotlib.pyplot as plt

#Pytorch libraries and modules
import torch

from torchsummary import summary

from torch.optim import Adam
from torch.utils.data import DataLoader

#torchvision for pre-trained models and augmentation

#for evaluating model

from sklearn.model_selection import train_test_split
import argparse


def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--CHECKPOINT_PATH", default = './model/UNet.pt',type=str)
    parser.add_argument('--img_path', default='./dataset_lungseg/images/', type=str)
    parser.add_argument('--mask_path', default='./dataset_lungseg/masks/', type= str)

    parser.add_argument('--BATCH_SIZE', default=8, type=int)
    parser.add_argument('--num_epochs', default= 50, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)

    opt = parser.parse_args()
    return opt



def dataloader():

    mask_path = './dataset_lungseg/masks/'
    img_path = './dataset_lungseg/images/'
    mask_list = os.listdir(mask_path)
    img_mask_list = [(mask_names.replace('_mask',''), mask_names) for mask_names in mask_list]

    train_list, test_list = train_test_split(img_mask_list, test_size = 0.2, random_state = 42) 
    train_list, val_list = train_test_split(train_list,test_size = 0.1, random_state = 42)
    # print(len(train_list), len(val_list), len(test_list)) # 506, 57, 141
    # print(train_list)

    image_t = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.5],  std = [0.5])    
    ])

    mask_t = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    train_set = LungDataset(train_list, img_path, mask_path, transform = (image_t, mask_t))
    val_set = LungDataset(val_list, img_path, mask_path, transform = (image_t, mask_t))
    test_set = LungDataset(test_list, img_path, mask_path, transform = (image_t, mask_t))

    loader ={
        'train' : DataLoader(
            train_set, 
            batch_size= 4,
            shuffle=True
        ),
        'val' : DataLoader(
            val_set, 
            batch_size=4,
            shuffle=True
        ),
        'test' : DataLoader(
            test_set, 
            batch_size=4,
            shuffle=True
        )
    }   
    return loader


def main():
    opt = get_opt()
    

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # criterion = BCEWithLogitsLoss()
    model = UNet_ResNet.to(device)
    optimizer = Adam(model.parameters(), opt.lr)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=14, gamma=0.1)
    loss_fn = DiceBCELoss()

    
    best_valid_loss = float("inf")

    torch.cuda.empty_cache()

    # train
    res = fit(model, dataloader()['train'], dataloader()['val'], optimizer, opt.num_epochs, loss_fn, calculate_metrics, opt.CHECKPOINT_PATH, device)
    
    # visualize loss, acc
    loss, val_loss = res['loss'], res['val_loss']
    acc, val_acc = res['acc'], res['val_acc']
    plot_acc_loss (loss, val_loss, acc, val_acc)

    # test
    with torch.no_grad():
        for x, y in dataloader()['val']:
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            y_pred = model(x)
            break

    pred = y_pred.cpu().numpy()
    ynum = y.cpu().numpy()

    pred = pred.reshape(len(pred), 224, 224)
    ynum = ynum.reshape(len(ynum), 224, 224)

    pred = pred > 0.5
    pred = np.array(pred, dtype=np.uint8)

    ynum = ynum > 0.5
    ynum = np.array(ynum, dtype=np.uint8)

    plt.imshow(pred[0], cmap='gray')
    plt.imshow(ynum[0], cmap='gray')





if __name__ == '__main__':
    main()
   

    