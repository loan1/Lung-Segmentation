

from Script.train import trainKFold
from Script.test import mainTest, load_np
from Script.Predict import predict
from Script.visualize import Testimshow
from Script.utils import seed_everything
import Script.models as models
import torch

if __name__ == '__main__':

    seed = 152022 # set seed value

    seed_everything(seed)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = models.list_modelUNet[6]
    # train KFold
    trainKFold(model)

    # test 
    # mainTest(cp_path='../model/rp17_5/ResNet18', device = device, path_np='../visualize/testResNet18', model = model)
    # images_np, masks_np, y_prect, y_arg = load_np(path = '../visualize/testResNet18')  
    # Testimshow(images_np, masks_np, y_prect, y_arg, '../visualize/testResNet18.png')

    # inference

