import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
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




# def IoU(outputs: torch.Tensor, labels: torch.Tensor,threshold=0.5):

#     SMOOTH = 1e-6
#     outputs = outputs.squeeze(1)
#     labels = labels.squeeze(1)
#     bin_out = torch.where(outputs > threshold,1,0).type(torch.int16)
#     labels = labels.type(torch.int16)
#     intersection = (bin_out & labels).float().sum((1,2))
#     union = (bin_out | labels).float().sum((1,2))
#     iou = (intersection + SMOOTH) / (union + SMOOTH)
#     thresholded = torch.clamp(20 * (iou - 0.5),0,10).ceil() / 10 
    
#     return thresholded.mean()

# def dice_loss(logits,true,eps=1e-7): 
#     num_classes = logits.shape[1]
    
#     if num_classes == 1: 
#         true_1_hot = torch.eye(num_classes + 1)[true.long().squeeze(0)]
#         print(true_1_hot)
#         true_1_hot = true_1_hot.permute(0,3,1,2).float()
#         true_1_hot_f = true_1_hot[:,0:1,:,:]
#         true_1_hot_s = true_1_hot[:,1:2,:,:]
#         true_1_hot = torch.cat([true_1_hot_s,true_1_hot_f],dim=1)
#         pos_prob = torch.sigmoid(logits)
#         neg_prob = 1 - pos_prob
#         probas = torch.cat([pos_prob,neg_prob],dim=1)
    
#     else: 
#         true_1_hot = torch.eye(num_classes)[true.long().squeeze(1)]
#         true_1_hot = true_1_hot.permute(0,3,1,2).float()
#         probas = F.softmax(logits,dim=1)
    
#     true_1_hot = true_1_hot.type(logits.type())
#     dims = (0,) + tuple(range(2,true.ndimension()))
#     intersection = torch.sum(probas * true_1_hot,dims)
#     cardinality = torch.sum(probas + true_1_hot,dims)
#     dice_loss = (2. * intersection / (cardinality + eps)).mean()

#     return (1 - dice_loss) 

# class ShapeAwareLoss(nn.Module):
#     def __init__(self):
#         super(ShapeAwareLoss, self).__init__()

#     def forward(self, predicted, target):
#         # Calculate the Fourier descriptors of the predicted and target shapes.
#         predicted = predicted.squeeze(1)
#         predicted_descriptors = torch.fft(predicted)
#         target_descriptors = torch.fft(target)

#         # Calculate the shape dissimilarity between the predicted and target shapes.
#         shape_dissimilarity = torch.mean((predicted_descriptors - target_descriptors)**2)

#         # Return the loss value.
#         return shape_dissimilarity
    

# def BCEWithLogitsLoss(output,target): 
#     bce = nn.BCEWithLogitsLoss()
#     return bce(output,target)


# def BCE(output,target): 
#     bce = BCE(output,target)
#     return bce(output,target)


# def BCE_dice_loss(output,target): 
#     bce = nn.BCEWithLogitsLoss()
#     return bce(output,target) + dice_loss(output,target)

# def dice_loss(y_true, y_pred, smooth=1.0):
#     y_true_f = y_true.view(-1)  # Flatten the tensor
#     y_pred_f = y_pred.view(-1)  # Flatten the tensor
#     intersection = torch.sum(y_true_f * y_pred_f)
#     return 1 - (2. * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)

# def BCE_dice_loss(output,target): 
#     bce = nn.BCEWithLogitsLoss()
#     return bce(output,target) + dice_loss(output,target)   

    



    