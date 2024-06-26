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
                
        return 1 - IoU
    
ALPHA = 0.5
BETA = 0.5

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
    
    

class TLoss(nn.Module):
    def __init__(
        self,
        image_size: int,
        device: torch.device,
        nu: float = 1.0,
        epsilon: float = 1e-8,
        reduction: str = "mean",
    ):
        """
        Implementation of the TLoss.

        Args:
            config: Configuration object for the loss.
            nu (float): Value of nu.
            epsilon (float): Value of epsilon.
            reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
                             'none': no reduction will be applied,
                             'mean': the sum of the output will be divided by the number of elements in the output,
                             'sum': the output will be summed.
        """
        super().__init__()
        self.D = torch.tensor(
            (image_size * image_size),
            dtype=torch.float,
            device=device,
        )
 
        self.lambdas = torch.ones(
            (image_size, image_size),
            dtype=torch.float,
            device=device,
        )
        self.nu = nn.Parameter(
            torch.tensor(nu, dtype=torch.float, device=device)
        )
        self.epsilon = torch.tensor(epsilon, dtype=torch.float, device=device)
        self.reduction = reduction

    def forward(
        self, input_tensor: torch.Tensor, target_tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            input_tensor (torch.Tensor): Model's prediction, size (B x W x H).
            target_tensor (torch.Tensor): Ground truth, size (B x W x H).

        Returns:
            torch.Tensor: Total loss value.
        """

        delta_i = input_tensor - target_tensor
        sum_nu_epsilon = torch.exp(self.nu) + self.epsilon
        first_term = -torch.lgamma((sum_nu_epsilon + self.D) / 2)
        second_term = torch.lgamma(sum_nu_epsilon / 2)
        third_term = -0.5 * torch.sum(self.lambdas + self.epsilon)
        fourth_term = (self.D / 2) * torch.log(torch.tensor(np.pi))
        fifth_term = (self.D / 2) * (self.nu + self.epsilon)

        delta_squared = torch.pow(delta_i, 2)
        lambdas_exp = torch.exp(self.lambdas + self.epsilon)
        numerator = delta_squared * lambdas_exp
        numerator = torch.sum(numerator, dim=(1, 2))

        fraction = numerator / sum_nu_epsilon
        sixth_term = ((sum_nu_epsilon + self.D) / 2) * torch.log(1 + fraction)

        total_losses = (
            first_term
            + second_term
            + third_term
            + fourth_term
            + fifth_term
            + sixth_term
        )

        if self.reduction == "mean":
            return total_losses.mean()
        elif self.reduction == "sum":
            return total_losses.sum()
        elif self.reduction == "none":
            return total_losses
        else:
            raise ValueError(
                f"The reduction method '{self.reduction}' is not implemented."
            )










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

    



    