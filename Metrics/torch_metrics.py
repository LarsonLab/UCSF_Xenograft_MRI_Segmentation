import numpy as np 
import torch 
import torchvision 
from torchvision import utils as vutils 
from torchvision import transforms 
from torch import nn 
from torch.nn import functional as F 
from torch.optim import SGD, Adam 
from statistics import mean 
import matplotlib.pyplot as plt 
import datetime

current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

SMOOTH = 1e-6

def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    
    intersection = torch.logical_and(outputs, labels).float().sum((1, 2))
    union = (outputs | labels).float().sum((1, 2))         # Will be zzero if both are 0
    
    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
    
    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
    
    return thresholded 

def IoU(outputs: torch.Tensor, labels: torch.Tensor,threshold=0.5):

    SMOOTH = 1e-6
    outputs = outputs.squeeze(1)
    labels = labels.squeeze(1)
    bin_out = torch.where(outputs > threshold,1,0).type(torch.int16)
    labels = labels.type(torch.int16)
    intersection = (bin_out & labels).float().sum((1,2))
    union = (bin_out | labels).float().sum((1,2))
    iou = (intersection + SMOOTH) / (union + SMOOTH)
    thresholded = torch.clamp(20 * (iou - 0.5),0,10).ceil() / 10 
    
    return thresholded.mean()

def dice_loss(logits,true,eps=1e-7): 
    num_classes = logits.shape[1]
    
    if num_classes == 1: 
        true_1_hot = torch.eye(num_classes + 1)[true.long().squeeze(1)]
        true_1_hot = true_1_hot.permute(0,3,1,2).float()
        true_1_hot_f = true_1_hot[:,0:1,:,:]
        true_1_hot_s = true_1_hot[:,1:2,:,:]
        true_1_hot = torch.cat([true_1_hot_s,true_1_hot_f],dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob,neg_prob],dim=1)
    
    else: 
        true_1_hot = torch.eye(num_classes)[true.long().squeeze(1)]
        true_1_hot = true_1_hot.permute(0,3,1,2).float()
        probas = F.softmax(logits,dim=1)
    
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2,true.ndimension()))
    intersection = torch.sum(probas * true_1_hot,dims)
    cardinality = torch.sum(probas + true_1_hot,dims)
    dice_loss = (2. * intersection / (cardinality + eps)).mean()

    return (1 - dice_loss) 

class ShapeAwareLoss(nn.Module):
    def __init__(self):
        super(ShapeAwareLoss, self).__init__()

    def forward(self, predicted, target):
        # Calculate the Fourier descriptors of the predicted and target shapes.
        predicted = predicted.squeeze(1)
        predicted_descriptors = torch.fft(predicted)
        target_descriptors = torch.fft(target)

        # Calculate the shape dissimilarity between the predicted and target shapes.
        shape_dissimilarity = torch.mean((predicted_descriptors - target_descriptors)**2)

        # Return the loss value.
        return shape_dissimilarity
    

def BCEWithLogitsLoss(output,target): 
    bce = nn.BCEWithLogitsLoss()
    return bce(output,target)


def BCE(output,target): 
    bce = BCE(output,target)
    return bce(output,target)


def BCE_dice_loss(output,target): 
    bce = nn.BCEWithLogitsLoss()
    return bce(output,target) + dice_loss(output,target)

def plot_training_scores(losses,train_scores,save_path): 
    fig, axes = plt.subplots(nrows=1,ncols=2,figsize=(5,5))
    axes[0].set_title('Train BCE Loss')
    axes.plot(range(len(losses)),losses)
    axes[1].set_title('IoU Score vs Training Step')
    axes[1].plot(range(len(train_scores)),train_scores)

    print(f'MEAN TRAIN IOU: {torch.mean(torch.tensor(train_scores))}')
    plt.savefig(f'{save_path}Training_Scores:{current_time}')

def plot_validation_scores(val_losses, val_scores,save_path): 

    fig, axs = plt.subplots(1,2,figsize=(7,7))
    axs[0].set_title('BCE Loss on Validation Set')
    axs[0].hist(val_losses)

    temp = [t.cpu().item() for t in val_scores]
    axs[1].set_title('IoU Scores on Validation Set')
    axs[1].hist(temp)
    axs[1].axvline(np.median(np.array(temp)),color='k',
                   linestyle='dashed',linewidth=1)
    print(f"MEAN VAL IOU: {mean(temp)}")
    plt.savefig(f'{save_path}Validation_Scores:{current_time}')
    





    



    