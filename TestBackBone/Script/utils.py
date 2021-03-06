# https://github.com/qubvel/segmentation_models.pytorch
# https://github.com/IlliaOvcharenko/lung-segmentation
# https://www.kaggle.com/pezhmansamadi/lung-segmentation-torch

import numpy as np
import torch
import time
# from sklearn.model_selection import KFold
import os
import random

#loss
import torch.nn as nn
import torch.nn.functional as F

# metrics
from sklearn.metrics import accuracy_score, jaccard_score, f1_score, recall_score, precision_score
from operator import add
import sys

class ComboLoss(nn.Module): #Dice + BCE + focal
    def __init__(self, weight=None, size_average=True):
        super(ComboLoss, self).__init__()

    def forward(self, inputs, targets, alpha = 0.8, gamma = 2,smooth=1):
        inputs = torch.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha*(1-BCE_EXP)**gamma*BCE
        
        Dice_BCE = BCE + dice_loss +focal_loss

        return Dice_BCE

def calculate_metrics(y_pred, y_true):
    """ Ground truth """
    y_true = y_true.cpu().numpy()
    y_true = y_true > 0.5
    y_true = y_true.astype(np.uint8)
    y_true = y_true.reshape(-1)

    """ Prediction """
    y_pred = y_pred.cpu().detach().numpy()
    y_pred = y_pred > 0.5 # True False False True 
    y_pred = y_pred.astype(np.uint8) # 0 1 1 0
    y_pred = y_pred.reshape(-1) # flatten

    jaccard = jaccard_score(y_true, y_pred) #(IoU)
    f1 = f1_score(y_true, y_pred) # Dice
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)

    return [jaccard, acc, f1, recall, precision]

def reset_weights(m):
    '''
    Try resetting model weights to avoid weight leakage.
    '''
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            # print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()

def train(model, loader, optimizer, scheduler, loss_fn, metric_fn, device):

    epoch_loss = 0.0
    metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0]
    steps = len(loader)
    
    model.train()

    for i, (x, y) in enumerate (loader):
        x = x.to(device)
        y = y.float().unsqueeze(1).to(device)

        optimizer.zero_grad()

        y_pred = model(x) # gpu cuda

        loss = loss_fn(y_pred, y) # comboloss: BCE dice focal
        loss.backward() # 
        
        score = metric_fn(y_pred, y)
        metrics_score = list(map(add, metrics_score, score))
        
        optimizer.step()
        learning_rate = optimizer.param_groups[0]['lr']     

        epoch_loss += loss.item()
        
        sys.stdout.flush()
        sys.stdout.write('\r Step: [%2d/%2d], loss: %.4f - acc: %.4f' % (i, steps, loss.item(), score[1]))
    scheduler.step()
    
    sys.stdout.write('\r')

    epoch_loss = epoch_loss/len(loader)
    
    epoch_jaccard = metrics_score[0]/len(loader)
    epoch_acc = metrics_score[1]/len(loader)
    epoch_dice = metrics_score[2] / len(loader)
    epoch_recall = metrics_score[3] / len(loader)
    epoch_precision = metrics_score[4] / len(loader)
    
    return epoch_loss, epoch_jaccard, epoch_dice, epoch_recall, epoch_precision, epoch_acc, learning_rate,  

def evaluate(model, loader, loss_fn, metric_fn, device):
    epoch_loss = 0.0
    metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0]

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            # y = y.to(device, dtype=torch.float32)
            # x = x.float().to(device)
            y = y.float().unsqueeze(1).to(device)
            # print(x)
            # print(y)

            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            
            score = metric_fn(y_pred, y)
            metrics_score = list(map(add, metrics_score, score))
            
            epoch_loss += loss.item()

        epoch_loss = epoch_loss / len(loader)
        
        epoch_jaccard = metrics_score[0] / len(loader)
        epoch_acc = metrics_score[1] / len(loader)
        epoch_f1 = metrics_score[2] / len(loader)
        epoch_recall = metrics_score[3] / len(loader)
        epoch_precision = metrics_score[4] / len(loader)
    
    return epoch_loss, epoch_jaccard, epoch_f1, epoch_acc, epoch_recall, epoch_precision

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def fit (model, train_dl, valid_dl, optimizer, scheduler, epochs, loss_fn, metric_fn, checkpoint_path, fold, device):
    """ fiting model to dataloaders, saving best weights and showing results """
    losses, val_losses, accs, val_accs = [], [], [], []
    jaccards, val_jaccards, f1s, recalls, precisions = [], [], [], [], []
    learning_rate =[]

    best_val_loss = float("inf")
    patience = 8 
    checkpoint_path = checkpoint_path.replace('.pt', str(fold) + '.pt')
    since = time.time()
    for epoch in range (epochs):
        ts = time.time()
        
        loss, jaccard, dice,recall,precision, acc,  lr = train(model, train_dl, optimizer, scheduler, loss_fn, metric_fn, device)
        val_loss, val_jaccard, f1, val_acc, recall, precision = evaluate(model, valid_dl, loss_fn, metric_fn, device)
        
        
        losses.append(loss)
        accs.append(acc)
        jaccards.append(jaccard)
        
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_jaccards.append(val_jaccard)

        f1s.append(dice)
        recalls.append(recall)
        precisions.append(precision)

        learning_rate.append(lr)
        
        te = time.time() 

        epoch_mins, epoch_secs = epoch_time(ts, te)
        
        # print ('Epoch [{}/{}], loss: {:.4f} - jaccard: {:.4f} - acc: {:.4f}  - val_loss: {:.4f} - val_jaccard: {:.4f} - val_acc: {:.4f}'.format (epoch + 1, epochs, loss, jaccard, acc, val_loss, val_jaccard, val_acc))
        print ('Epoch [{}/{}], loss: {:.4f} - jaccard: {:.4f} - acc: {:.4f} '.format (epoch + 1, epochs, loss, jaccard, acc))
        print ('val_loss: {:.4f} - val_jaccard: {:.4f} - val_acc: {:.4f} - val_f1: {:.4f} - val_recall: {:.4f} - val_precision: {:.4f}'.format (val_loss, val_jaccard, val_acc, f1, recall, precision))
        print(f'Time: {epoch_mins}m {epoch_secs}s')
    
        period = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(period // 60, period % 60))
        if val_loss < best_val_loss:
            count = 0
            data_str = f"===> Valid loss improved from {best_val_loss:2.4f} to {val_loss:2.4f}. Saving checkpoint: {checkpoint_path}"
            print(data_str)
            best_val_loss = val_loss
            # save_checkpoint(model.state_dict(), checkpoint_path)
            torch.save(model.state_dict(), checkpoint_path) #save checkpoint
        else:
            count += 1
            # print('count = ',count)
            if count >= patience:
                print('Early stopping!')
                return dict(loss = losses, val_loss = val_losses, acc = accs, val_acc = val_accs, jaccard = jaccards, val_jaccard = val_jaccards, learning_rate = learning_rate)



    return dict(loss = losses, val_loss = val_losses, acc = accs, val_acc = val_accs, jaccard = jaccards, val_jaccard = val_jaccards, learning_rate = learning_rate)

def seed_everything(seed):        

    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed) # set python seed

    np.random.seed(seed) # seed the global NumPy RNG

    torch.manual_seed(seed) # seed the RNG for all devices (both CPU and CUDA):
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
