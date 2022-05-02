
import matplotlib.pyplot as plt
import numpy as np


def imshow(input):
    """ Imshow for Tensor"""
    print(input.shape)
    # input = input.numpy(dtype=np.float32).transpose((2, 3, 1, 0))
    input = input.numpy().transpose((1, 2, 0))
    print(input.shape)
    input = input*0.5 + 0.5
    input = np.squeeze(input)
    # print(input)
    input = np.clip(input, 0, 1)
    plt.imshow(input, cmap = 'gray')
    plt.show()


def plot_acc_loss (loss, val_loss, acc, val_acc, path):
    """ plot training and validation loss and accuracy """
    plt.figure (figsize = (12, 4))
    plt.subplot (1, 2, 1)
    plt.plot (range (len (loss)), loss, 'b-', label = 'Training')
    plt.plot (range (len (loss)), val_loss, 'bo-', label = 'Validation')
    plt.xlabel ('Epochs')
    plt.ylabel ('Loss')
    plt.title ('Loss')
    plt.legend ()

    plt.subplot (1, 2, 2)
    plt.plot (range (len (acc)), acc, 'b-', label = 'Training')
    plt.plot (range (len (acc)), val_acc, 'bo-', label = 'Validation')
    plt.xlabel ('Epochs')
    plt.ylabel ('accuracy')
    plt.title ('Accuracy')
    plt.legend ()

    plt.savefig(path)
    
    plt.show ()
    # plt.savefig(path)

def plot_LR(LR, path):
    plt.plot(range(len(LR)), LR, 'r', label = 'Learning rate')
    plt.xlabel('Epochs')
    plt.ylabel('Learning_Rate')
    plt.title('Learning Rate')
    plt.legend ()
    plt.savefig(path)
    plt.show()
##################################################################################################
def Testimshow(original,true,pred, mean, path):

    for i in range(51):

        plt.figure (figsize = (15, 20))

        for idx in range(4):

            plt.subplot (4,8,idx*8 +1)
            plt.imshow(original[i][idx], cmap='gray')
            # plt.title('Original Image')

            plt.subplot (4,8,idx*8 +2)
            plt.imshow(true[i][idx], cmap='gray')
            # plt.title('True Mask')

            plt.subplot (4,8,idx*8 +3)
            plt.imshow(pred[0][i][idx], cmap='gray')
            # plt.title('Predict Mask')

            plt.subplot (4,8,idx*8 +4)
            plt.imshow(pred[1][i][idx], cmap='gray')
            # plt.title('Predict Mask')

            plt.subplot (4,8,idx*8 +5)
            plt.imshow(pred[2][i][idx], cmap='gray')
            # plt.title('Predict Mask')

            plt.subplot (4,8,idx*8 +6)
            plt.imshow(pred[3][i][idx], cmap='gray')
            # plt.title('Predict Mask')


            plt.subplot (4,8,idx*8 +7)
            plt.imshow(pred[4][i][idx], cmap='gray')
            # plt.title('Predict Mask')

            plt.subplot (4,8,idx*8 +8)
            # print(mean.shape)
            plt.imshow(mean[i][idx], cmap='gray')
            # plt.title('Predict Mean')
            
        plt.savefig(path.replace('.png', '_' +str(i) + '.png'))
        # plt.title('Original ---- True ---- Predict ---- Predict ---- Predict ---- Predict ---- Predict ---- Mean')
        # plt.show()    
