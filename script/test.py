from utils import UNet_ResNet 
from dataset import *

import albumentations as A
import os
from torch.utils.data import DataLoader
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

def dataloader():

    mask_path = '../dataset_lungseg/test/lungVAE-masks/'
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

    # with open('../dataset_lungseg/test/mask.txt','w') as f:
    #     for i in range(len(mask_list)):
    #         f.write(str(mask_list[i])+ '\n') 

    image_t = A.Compose([
        A.Resize((224, 224)),
        A.ToTensor(),
        A.Normalize(mean = [0.5],  std = [0.5])    
    ])

    mask_t = A.Compose([
        A.Resize((224,224)),
        A.ToTensor()
    ])
    test_set = LungDataset(img_mask_list, img_path, mask_path, transform = (image_t, mask_t))

    loader ={

        'test' : DataLoader(
            test_set, 
            batch_size=4,
            shuffle=True
        )
    }   
    # print(len(loader['test']))
    return loader

def test(dataloader, device, model):
   
    with torch.no_grad():
        image, y_true, y_pred = [], [], []
        model.eval()
        for x, y in dataloader:
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            pred = model(x)

            pred = pred.cpu().numpy() # mask output 
            ynum = y.cpu().numpy()  # mask label
            # x = x.cpu().numpy() 
            # print('len ', len(x))
            pred = pred.reshape(len(pred), 224, 224)
            ynum = ynum.reshape(len(ynum), 224, 224)
            # x = x.reshape(len(x), 224, 224)


            pred = pred > 0.5
            pred = np.array(pred, dtype=np.uint8)

            ynum = ynum > 0.5
            ynum = np.array(ynum, dtype=np.uint8)   

            # x = x > 0.5
            # x = np.array(x, dtype=np.uint8)

            y_true.append(ynum)    
            y_pred.append(pred)
        # print(pred.shape)

            # print(x.shape)
            x = x.cpu().numpy()
            # print('len x', len(x))
            x = x.reshape(len(x), 224, 224)
            # print(x.shape)
            x = x*0.5 + 0.5
            x = np.squeeze(x)
            # # print(input)
            x = np.clip(x, 0, 1)
            image.append(x)

    return image, y_true, y_pred 

def predict(x, model, device):
    model.eval()
    with torch.no_grad():
        x = x.to(device, dtype=torch.float32)
        # y = y.to(device, dtype=torch.float32)

        # output= model(x)
        # _,pred = torch.max(output, 1)

        y_pred = model(x)
        pred = y_pred.cpu().numpy() # mask output 
        # ynum = y.cpu().numpy()  # mask label

        pred = pred.reshape(len(pred), 224, 224)
        # ynum = ynum.reshape(len(ynum), 224, 224)

        pred = pred > 0.5
        pred = np.array(pred, dtype=np.uint8)

        # ynum = ynum > 0.5
        # ynum = np.array(ynum, dtype=np.uint8)   
        x = x.cpu().numpy()
        # print('len x', len(x))
        x = x.reshape(len(x), 224, 224)
        # print(x.shape)
        x = x*0.5 + 0.5
        x = np.squeeze(x)
        # # print(input)
        x = np.clip(x, 0, 1)

    return x, pred
    

# image, mask = next(iter(dataloader()['test']))
# image1 = image[0][None,:,:,:]
# print(image1.shape)

def visualize (y_true, y_pred, classes = 1):
    cnf_matrix = confusion_matrix(y_true, y_pred)
    fix, ax = plt.subplots(figsize = (10,10))
    disp = ConfusionMatrixDisplay(confusion_matrix = cnf_matrix, display_labels = classes)
    disp.plot(include_values = True, cmap = 'Blues', ax = ax, xticks_rotation = 'vertical')
    plt.savefig('../report/FTVGG19bn/MatrixFTVGG19bn1.png')

# def main():
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#     model = UNet_ResNet.to(device)
    
#     checkpoint = torch.load('../model/UNet.pt')
#     model.load_state_dict(checkpoint)

#     x, y_true, y_pred = test(dataloader()['test'], device, model)
      
#     print(len(dataloader()['test']))
#     print(y_true[1].shape)

#     image, mask = next(iter(dataloader()['test']))

#     y_true, y_pred = predict(image, mask, model, device)
#     print(y)
#     print(len(y_pred))

#     y_true =[(4,224,224),...]

#     print(x.shape)

#     print(y_pred[0][3].shape)
#     plt.figure (figsize = (15, 20))
#     for idx in range(4):

#         plt.subplot (4,3,idx*2 +idx +1)
#         plt.imshow(x[0][idx], cmap='gray')
#         plt.title('Original Image')

#         plt.subplot (4,3,idx*2 +idx +2)
#         plt.imshow(y_true[0][idx], cmap='gray')
#         plt.title('True Mask')

#         plt.subplot (4,3,idx*2+idx +3)
#         plt.imshow(y_pred[0][idx], cmap='gray')
#         plt.title('Predict Mask')

#     plt.show()

#     # accuracy = accuracy_score(y_true, y_pred)
#     # print(accuracy)

def main1():
    image_tfm = A.Compose([
        A.Resize((224, 224)),
        A.ToTensor(),
        A.Normalize(mean = [0.5],  std = [0.5]) 
        

    ])

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = UNet_ResNet.to(device)
    
    checkpoint = torch.load('../model/UNet.pt')
    model.load_state_dict(checkpoint)

    img = Image.open('../dataset_lungseg/predict/1dad3414-88c9-4c56-af5d-3a1488af452c.png').convert('L')
    img = image_tfm(img) 
    img = img.unsqueeze(0)
    # print(img.shape)

    x, y_pred = predict(img, model, device)

    image, mask = next(iter(dataloader()['test']))

    # print(x.shape)

    # print(y_pred[0][3].shape)
    plt.figure (figsize = (15, 20))

    plt.subplot (1,2,1)
    plt.imshow(x, cmap='gray')
    plt.title('Original Image')

    plt.subplot (1,2,2)
    plt.imshow(y_pred[0], cmap='gray')
    plt.title('Predict Mask')

    plt.show()

    # accuracy = accuracy_score(y_true, y_pred)
    # print(accuracy)
if __name__ == '__main__':
    main1()

    


