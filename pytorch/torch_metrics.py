import numpy as np 
import torch 
import torchvision 
from torchvision import utils as vutils 
from torchvision import transforms 
from torch import nn 
from torch.nn import functional as F 
from torch.optim import SGD, Adam 
from statistics import mean 



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



    



    