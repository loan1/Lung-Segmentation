# link dataset
# https://www.kaggle.com/andyczhao/covidx-cxr2/code
from Script.dataset_custom import DatasetPredict

import cv2
from skimage import morphology
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
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

    dataNegative = DatasetPredict(data_dir + 'Negative/', transform = transfms)
    dataPositive = DatasetPredict(data_dir + 'Positive/', transform = transfms)

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
        image, y_predict = [], []
        for x, name in tqdm(dataloader):
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

def mainPredict(model):
    # cp_list = ['../../model/UNetResNet18Gauss/UNet0.pt', '../../model/UNetResNet18Gauss/UNet1.pt', '../../model/UNetResNet18Gauss/UNet2.pt', '../../model/UNetResNet18Gauss/UNet3.pt', '../../model/UNetResNet18Gauss/UNet4.pt']
    cp_list = ['../../model/VGG11_bn/UNet0.pt', '../../model/VGG11_bn/UNet1.pt', '../../model/VGG11_bn/UNet2.pt', '../../model/VGG11_bn/UNet3.pt', '../../model/VGG11_bn/UNet4.pt']
    y = []
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # modelUNet = UNet_ResNet18.to(device)
    modelUNet = model.to(device)
    
    for fold in tqdm(range(len(cp_list))):
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
        y_mean[idx] = y_mean[idx] > 0.1
        y_mean[idx] = y_mean[idx].astype(np.uint8)

    # print('x.shape ',x.shape)
    # # print('y[1].shape ',y[1].shape)
    # print('y_mean.shape ',y_mean.shape)


    # imshowPre(x, y, y_mean, '../../visualize/InferPositive/InferResNet18GaussPreFULL/InferResNet18.png')

    # np.save('../../visualize/InferResNet18GaussPre_NegFULL/images.npy',x)
    # np.save('../../visualize/InferResNet18GaussPre_NegFULL/y_predict_5folds.npy',y)
    # np.save('../../visualize/InferResNet18GaussPre_NegFULL/y_predict_mean_5folds.npy',y_mean)
    
    np.save('../../visualize/InferVGG11_bn_PosFULL/images.npy',x)
    np.save('../../visualize/InferVGG11_bn_PosFULL/y_predict_5folds.npy',y)
    np.save('../../visualize/InferVGG11_bn_PosFULL/y_predict_mean_5folds.npy',y_mean)

    # plt.show()
def postprocess(img):
    # img = cv2.imread(img_name, 0)
    areas = []
    img = img.astype("uint8")
    blur = cv2.GaussianBlur(img, (3,3), 0) #làm mờ ảnh
    _, thresh = cv2.threshold(blur, 0,1, cv2.THRESH_BINARY + cv2.THRESH_OTSU) #nhị phân hóa ảnh

    contours, hierarchy = cv2.findContours(thresh.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) #tính contours

    for i in range(len(contours)):
        areas.append(cv2.contourArea(contours[i]))
    areas = np.sort(areas)[::-1]

    thresh = thresh.astype(bool)
    if len(contours) > 1:
        thresh = morphology.remove_small_objects(thresh.copy(),areas[1])
    if len(contours) > 2 :
        thresh = morphology.remove_small_holes(thresh, areas[2])
    
    return thresh


if __name__ == '__main__':

    # mainPredict()
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
    # img, name = next(iter(dataloaderPre()['Positive']))
    # # print(name)
    # for i in range(4):
    #     plt.imshow(img[i][0], cmap='gray')
    #     plt.title(name[i])
    #     plt.show()
    # print(img)
    # for i in range(len(img)):
    #     plt.imshow(img[i][0], cmap='gray')
    #     plt.show()


    #######################################################################################################
    ###################################
    ### TAO FILE TXT LUU TEN ANH #####
    ##################################
    # list_name =[]
    # for img,name in tqdm(dataloaderPre()['Positive']):
    #     list_name.append(name)
    # # print(len(list_name)) # 4123
    # # print(list_name)

    # res = []
    # for i in tqdm(range(len(list_name)-1)):
    #     for idx in range(4):
    #         res.append(list_name[i][idx])
    
    # for i in range(2): #xu li phan le trong batch cuoi
    #     res.append(list_name[len(list_name)-1][i])

    # # print(res)

    # np.savetxt('../../dataset_lungseg/predict/filenamePos.txt', res, fmt = '%s')
    ########################################################################################################

          # print(y_mean.shape) # y_mean = 4123 x 4 x 256 x 256
    
    # ######################################################################################

    images_np = np.load('../../visualize/InferVGG11_bn_PosFULL/images.npy', allow_pickle=True)
    y = np.load('../../visualize/InferVGG11_bn_PosFULL/y_predict_5folds.npy', allow_pickle=True)
    y_mean = np.load('../../visualize/InferVGG11_bn_PosFULL/y_predict_mean_5folds.npy', allow_pickle=True)

    list_name = np.loadtxt('../../dataset_lungseg/predict/filenamePos.txt', dtype = list)

    for i in tqdm(range(len(dataloaderPre()['Positive']))):   # 
        for idx in range(4):

            ret = postprocess(y_mean[i][idx])           
            plt.imsave('../../dataset_lungseg/predict/PosMaskVGG11/' + list_name[i*4+idx], ret)

    # imshow(images_np, y, y_mean, ret'../../visualizeTestResNet18/testResNet1803.png')

    plt.close('all')





    