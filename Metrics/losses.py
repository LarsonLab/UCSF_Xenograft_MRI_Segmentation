import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd.function import Function
from torch import Tensor, einsum 
from torch.nn import PoissonNLLLoss
import six 
from typing import List, cast 
from skimage.filters import sobel 
from scipy import ndimage as ndi
from typing import List, cast 
import torch 
import numpy as np 
from torch import Tensor, einsum 
from Metrics.utils import one_hot_encode
# import keras
# import keras.backend as K

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice


class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE
    
class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
        #print(IoU)
                
        return 1 - IoU
    
ALPHA = 0.5
BETA = 0.5
THETA = 0.5
GAMMA = 0.5
IOTA = 0.5
MU = 0.5

class TverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(TverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=ALPHA, beta=BETA):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
       
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        
        return 1 - Tversky
    
class DiceIouLoss(nn.Module):

    def __init__(self,dice_percent=ALPHA,iou_percent=BETA):
        super(DiceIouLoss,self).__init__()
        self.dice_percent = dice_percent
        self.iou_percent = iou_percent 
        self.dice_loss = DiceLoss()
        self.iou_loss = IoULoss()

    def forward(self,inputs,targets):
        dice_loss = self.dice_loss(inputs,targets)
        iou_loss = self.iou_loss(inputs,targets)
        combined_loss = (self.dice_percent * dice_loss) +  (self.iou_percent * iou_loss)
        return combined_loss 
    
class TverskyIoULoss(nn.Module):

    def __init__(self,tversky_percent=THETA,iou_percent=GAMMA):
        super(TverskyIoULoss,self).__init__()
        self.tverksy_percent = tversky_percent
        self.iou_percent = iou_percent 
        self.tversky_loss = TverskyLoss()
        self.iou_loss = IoULoss()

    def forward(self,inputs,targets):
        tversky_loss = self.tversky_loss(inputs,targets)
        iou_loss = self.iou_loss(inputs,targets)
        combined_loss = (tversky_loss * self.tverksy_percent) + (iou_loss * self.iou_percent)
        return combined_loss 

    

class BoundaryLoss(nn.Module):
    """Boundary Loss proposed in:
    Alexey Bokhovkin et al., Boundary Loss for Remote Sensing Imagery Semantic Segmentation
    https://arxiv.org/abs/1905.07852
    """

    def __init__(self, theta0=3, theta=5):
        super().__init__()

        self.theta0 = theta0
        self.theta = theta

    def forward(self, pred, gt):
        """
        Input:
            - pred: the output from model (before softmax)
                    shape (N, C, H, W)
            - gt: ground truth map
                    shape (N, H, w)
        Return:
            - boundary loss, averaged over mini-bathc
        """

        n, c, h, w = pred.shape

        # softmax so that predicted map can be distributed in [0, 1]
        pred = torch.sigmoid(pred)

        # one-hot vector of ground truth
        #one_hot_gt = one_hot_encode(gt, c)
        gt = gt.float()

        # boundary map
        gt_b = F.max_pool2d(
            1 - gt, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        gt_b -= 1 - gt

        pred_b = F.max_pool2d(
            1 - pred, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        pred_b -= 1 - pred

        # extended boundary map
        gt_b_ext = F.max_pool2d(
            gt_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)

        pred_b_ext = F.max_pool2d(
            pred_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)

        # reshape
        gt_b = gt_b.view(n, c, -1)
        pred_b = pred_b.view(n, c, -1)
        gt_b_ext = gt_b_ext.view(n, c, -1)
        pred_b_ext = pred_b_ext.view(n, c, -1)

        # Precision, Recall
        P = torch.sum(pred_b * gt_b_ext, dim=2) / (torch.sum(pred_b, dim=2) + 1e-7)
        R = torch.sum(pred_b_ext * gt_b, dim=2) / (torch.sum(gt_b, dim=2) + 1e-7)

        # Boundary F1 Score
        BF1 = 2 * P * R / (P + R + 1e-7)

        # summing BF1 Score for each class and average over mini-batch
        loss = torch.mean(1 - BF1)

        return loss

    

class BoundaryIoULoss(nn.Module):

    def __init__(self,boundary_percent=IOTA,iou_percent=MU):
        super(BoundaryIoULoss,self).__init__()
        self.boundary_percent = boundary_percent 
        self.iou_percent = iou_percent 
        self.boundary_loss = BoundaryLoss()
        self.iou_loss = IoULoss()

    def forward(self,input,target):
        boundary_loss = self.boundary_loss(input,target)
        iou_loss = self.iou_loss(input,target)
        combined_loss = (boundary_loss * self.boundary_percent) + (iou_loss * self.iou_percent)

        return combined_loss 
    
def is_binary_prob(t:Tensor,axis=1) -> bool:
    return ((t>=0) & (t <= 1)).all()