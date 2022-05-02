# https://www.kaggle.com/pezhmansamadi/lung-segmentation-torch

import os
import Script.utils as utils
from Script.dataset_custom import LungDataset



#importing the libraries

# https://drive.google.com/file/d/1ffbbyoPf-I3Y0iGbBahXpWqYdGd7xxQQ/view 

# import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
#for reading and displaying images


#Pytorch libraries and modules
import torch

from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader

from sklearn.model_selection import KFold
import argparse

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--CHECKPOINT_PATH", default = '../model/UNet/UNet.pt',type=str)
    parser.add_argument('--img_path', default='../dataset_lungseg/images/', type=str)
    parser.add_argument('--mask_path', default='../dataset_lungseg/masks/', type= str)

    parser.add_argument('--BATCH_SIZE', default=8, type=int)
    parser.add_argument('--num_epochs', default= 50, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)

    opt = parser.parse_args()
    return opt

def datasetKfold(kfold):

    mask_path = '../dataset_lungseg/masks/'
    mask_list = os.listdir(mask_path)
    img_mask_list = [(mask_names.replace('_mask',''), mask_names) for mask_names in mask_list]
    
    idx = kfold.split(img_mask_list) 
    return idx, img_mask_list

def get_item_to_idx(idx_list,image_list):
  train_list = [image_list[i] for i in idx_list]
  return train_list

def trainKFold(model):
    opt = get_opt()

    aug = A.Compose([
        A.Resize(512, 512), 
        A.HorizontalFlip(p=0.5),
        A.OneOf([
            A.RandomGamma(),# lấy hằng số gamma áp dụng cho BrightnessContrast
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),  # phép biến đổi co giãn
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.9),    # thay đổi độ sáng, độ tương phản        
        ], p = 0.3),
        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),     
        A.Normalize(mean = [0.5],  std = [0.5]),
        ToTensorV2()   
    ])

    transfm = A.Compose([
        A.Resize(512,512),
        A.Normalize(mean = [0.5],  std = [0.5]),
        ToTensorV2()
    ])

    k_folds = 5
    # For fold results
    results = {}
    # Set fixed random number seed
    torch.manual_seed(42)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    loss_fn = utils.ComboLoss()

    torch.cuda.empty_cache()

    # train and validation

    ###########################
    ############K-FOLD#########
    ###########################
    #define the K-fold Cross Validator
    kfold = KFold(n_splits=k_folds, shuffle=True)
    idx, img_mask_list = datasetKfold(kfold)

    # print(next(iter(idx)))
    #Start print
    print('=======================================')

    for fold, (train_ids, test_ids) in enumerate(idx): #

        #Print
        print(f'FOLD {fold}')
        print('-------------------------')
        train_ids = get_item_to_idx(train_ids, img_mask_list)
        test_ids  = get_item_to_idx(test_ids, img_mask_list)

        train_subsampler = LungDataset(train_ids,opt.img_path, opt.mask_path,aug)  
        test_subsampler = LungDataset(test_ids,opt.img_path, opt.mask_path,transfm)

        trainloader = DataLoader(
            train_subsampler,
            batch_size=8,
            shuffle=True
        )

        testloader = DataLoader(
            test_subsampler,
            batch_size=8,
            shuffle=True
        )
        model = model.to(device)

        model.apply(utils.reset_weights)
        optimizer = Adam(model.parameters(), opt.lr)

        scheduler3 = lr_scheduler.CosineAnnealingLR(optimizer, T_max = 10)

        loss_fn = utils.ComboLoss()
        
        res = utils.fit(model, trainloader, testloader, optimizer, scheduler3, opt.num_epochs, loss_fn, utils.calculate_metrics, opt.CHECKPOINT_PATH, fold, device)
    
        # visualize loss, acc
        loss, val_loss = res['loss'], res['val_loss']
        acc, val_acc = res['acc'], res['val_acc']

        Accuracy = 0
        for k in range(len(val_acc)):
            Accuracy +=  val_acc[k]
        
        # Print accuracy
        print ('Accuracy for fold %d: %f %%' % (fold, 100* Accuracy/len(val_acc)))
        print('--------------------------------')
        results[fold] = 100* Accuracy/len(val_acc)

      # Print fold results
    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS MODEL UNetDenseNet121')
    print('--------------------------------')
    sum = 0.0
    for key, value in results.items():
        print(f'Fold {key}: {value} %')
        sum += value
    print(f'Average: {sum/len(results.items())} %')



    