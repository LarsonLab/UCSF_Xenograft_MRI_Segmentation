import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from keras.losses import BinaryCrossentropy
import torch 
import datetime
import os
from torch import nn 

current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

smooth = 1e-5
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (1 - (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))

def IoU(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f)+K.sum(y_pred_f)-intersection
    return (intersection + smooth)/(union + smooth)

SMOOTH = 1e-6


def weighted_bce(y_true, y_pred):
    y_true_temp = K.flatten(y_true)
    y_pred_temp = K.flatten(y_pred)

    indices_ones = tf.where(y_true_temp)
    indices_zeros = tf.where(tf.equal(y_true_temp, 0))
    
    y_pred_ones = tf.gather(y_pred_temp, indices = indices_ones)
    y_pred_zeros = tf.gather(y_pred_temp, indices = indices_zeros)
    y_pred_ones = tf.reshape(y_pred_ones, [-1])
    y_pred_zeros = tf.reshape(y_pred_zeros, [-1])

    y_true_ones = tf.gather(y_true_temp, indices = indices_ones)
    y_true_zeros = tf.gather(y_true_temp, indices = indices_zeros)

    ones_size = tf.size(y_true_ones)
    zeros_size = tf.size(y_true_zeros)

    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    bce_ones = bce(y_true_ones, y_pred_ones)
    bce_zeros = bce(y_true_zeros, y_pred_zeros)
    bce_ones_score = K.mean(bce_ones)
    bce_zeros_score = K.mean(bce_zeros)

    if ones_size == 0:
        bce_balanced = bce_zeros_score
    elif zeros_size == 0:
        bce_balanced = bce_ones_score
    else:
        bce_balanced = (bce_ones_score + bce_zeros_score) / 2
    return bce_balanced

def composite_loss(y_true, y_pred):
    BCE_weight = 0.25 #change between 0.10 -.90
    dl_weight = 0.75

    # bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    # bce = tf.keras.losses.BinaryFocalCrossentropy(apply_class_balancing=True,
    #                                               alpha = 0.25,#14.028340861846038/(0.5184797236078723+14.028340861846038),
    #                                               gamma=0)
    # adjust bce to get value for both true and false pixels
    # bce_score = bce(y_true, y_pred)

    bce_score = weighted_bce(y_true, y_pred)
    dl = dice_loss(y_true, y_pred)
    loss = BCE_weight * bce_score + dl_weight * dl
    return loss

# class ImageSave(Callback):
#   def __init__(self,model,val_data,output_dir,save_freq=100):
#      self.model = model
#      self.val_data = val_data
#      self.output_dir = output_dir
#      self.save_freq = save_freq

#   def epoch(self,epoch):
#     if epoch % self.save_freq == 0:
#         self.save_images(epoch)

#   def save_images(self,epoch):
#     predications = self.model.predict(self.val_data)

def plot_training_scores(losses,train_scores,save_path): 
    fig, axes = plt.subplots(nrows=1,ncols=2,figsize=(5,5))
    axes[0].set_title('Train BCE Loss')
    axes.plot(range(len(losses)),losses)
    axes[1].set_title('IoU Score vs Training Step')
    axes[1].plot(range(len(train_scores)),train_scores)

    print(f'MEAN TRAIN IOU: {torch.mean(torch.tensor(train_scores))}')
    plt.savefig(f'{save_path}Training_Scores:{current_time}')

    


    
          
