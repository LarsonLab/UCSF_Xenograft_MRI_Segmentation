import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Input, Dropout, Add, Activation, UpSampling2D,  Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as K
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.metrics import Precision, Recall, AUC, Accuracy
from keras.losses import BinaryCrossentropy

# Optimizers
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.optimizers import SGD

# Schedulers
from schedulers import LinearDecay
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers.schedules import CosineDecay
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts

##
import os

K.set_image_data_format('channels_last')

# smooth = 1
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
    BCE_weight = 0.25
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

"""Recurrent Layer"""
def rec_layer(layer, filters):
    reconv1 = Conv2D(filters, (3, 3), activation='relu', padding='same')(layer)
    layer_add = Conv2D(filters, kernel_size=(1, 1), padding='same')(layer)
    add_conv1 = Add()([reconv1,layer_add])
    #drop_inter = Dropout(0.3)(reconc1)
    reconv1 = Conv2D(filters, (3, 3), activation='relu', padding='same')(add_conv1)
    add_conv2 = Add()([reconv1,layer_add])
    reconv1 = Conv2D(filters, (3, 3), activation='relu', padding='same')(add_conv2)
    return reconv1
    
pr_metric = AUC(curve='PR', num_thresholds=10, name = 'pr_auc') 
roc_metric = AUC(name = 'auc')
METRICS = [dice_loss,
    dice_coef,
    Precision(name='precision'),
    Recall(name='recall'),
    pr_metric,
    roc_metric,
    IoU,
    BinaryCrossentropy(name='binaryCE'),
    weighted_bce
]

########## Initialization of Parameters #######################
# original
# image_row = 128
# image_col = 128
# image_depth = 1

# new
image_row = 192
image_col = 192
image_depth = 1

def r2udensenet(optimizer = 'Adam', learning_rate = 1e-5, lr_scheduler = 'None', decay_steps = 1932 / 2 * 5):
    inputs = Input((image_row, image_col, image_depth))
    conv1 = rec_layer(inputs,32)
    conv1 = rec_layer(conv1,32)
    conv1add = Conv2D(32, kernel_size=(1, 1), padding='same')(inputs)
    add1 = Add()([conv1add, conv1])
    dense1 = concatenate([add1, conv1], axis=3)
    pool1 = MaxPooling2D(pool_size=(2, 2))(dense1)

    conv2 = rec_layer(pool1, 64)
    conv2 = rec_layer(conv2, 64)
    conv2add = Conv2D(64, kernel_size=(1, 1), padding='same')(pool1)
    add2 = Add()([conv2add, conv2])
    dense2 = concatenate([add2, conv2], axis=3)
    pool2 = MaxPooling2D(pool_size=(2, 2))(dense2)

    conv3 = rec_layer(pool2, 128)
    conv3 = rec_layer(conv3, 128)
    conv3add = Conv2D(128, kernel_size=(1, 1), padding='same')(pool2)
    add3 = Add()([conv3add, conv3])
    dense3 = concatenate([add3, conv3], axis=3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(dense3)

    conv4 = rec_layer(pool3, 256)
    conv4 = rec_layer(conv4, 256)
    conv4add = Conv2D(256, kernel_size=(1, 1), padding='same')(pool3)
    add4 = Add()([conv4add, conv4])
    dense4 = concatenate([add4, conv4], axis=3)
    drop4 = Dropout(0.5)(dense4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(dense4)

    conv5 = rec_layer(pool4, 512)
    conv5 = rec_layer(conv5, 512)
    conv5add = Conv2D(512, kernel_size=(1, 1), padding='same')(pool4)
    add5 = Add()([conv5add, conv5])
    dense5 = concatenate([add5, conv5], axis=3)
    drop5 = Dropout(0.5)(dense5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(dense5), conv4], axis=3)
    conv6 = rec_layer(up6, 256)
    conv6 = rec_layer(conv6, 256)
    conv6add = Conv2D(256, kernel_size=(1, 1), padding='same')(up6)
    add6 = Add()([conv6add, conv6])
    dense6 = concatenate([add6, conv6], axis=3)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(dense6), conv3], axis=3)
    conv7 = rec_layer(up7, 128)
    conv7 = rec_layer(conv7, 128)
    conv7add = Conv2D(128, kernel_size=(1, 1), padding='same')(up7)
    add7 = Add()([conv7add, conv7])
    dense7 = concatenate([add7, conv7], axis=3)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(dense7), conv2], axis=3)
    conv8 = rec_layer(up8, 64)
    conv8 = rec_layer(conv8, 64)
    conv8add = Conv2D(64, kernel_size=(1, 1), padding='same')(up8)
    add8 = Add()([conv8add, conv8])
    dense8 = concatenate([add8, conv8], axis=3)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(dense8), conv1], axis=3)
    conv9 = rec_layer(up9, 64)
    conv9 = rec_layer(conv9, 64)
    conv9add = Conv2D(64, kernel_size=(1, 1), padding='same')(up9)
    add9 = Add()([conv9add, conv9])
    dense9 = concatenate([add9, conv9], axis=3)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(dense9)
    model = Model(inputs=[inputs], outputs=[conv10])

    # Show model summary
    # model.summary()
    
    # For testing schedulers
    print(f"decay steps: {decay_steps}")
    if(lr_scheduler == 'Linear'):
        total_steps = decay_steps
        print(f"decay total_steps: {total_steps}")
        scheduler = LinearDecay(initial_learning_rate=learning_rate, final_learning_rate=0, total_steps=total_steps)
    elif(lr_scheduler == 'ExponentialDecay'):
        scheduler = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=learning_rate, decay_steps=decay_steps, decay_rate=0.96)
    elif(lr_scheduler == 'CosineDecay'):
        total_steps = decay_steps
        print(f"decay total_steps: {total_steps}")
        scheduler = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=learning_rate, decay_steps=total_steps)
    elif(lr_scheduler == 'CosineDecayRestarts'):
        scheduler = tf.keras.optimizers.schedules.CosineDecayRestarts(initial_learning_rate=learning_rate, first_decay_steps=decay_steps)
    else:
        print("### Using No Learning Rate Scheduler ###")
        scheduler = learning_rate
        # model.compile(optimizer=Adam(learning_rate=learning_rate), loss=composite_loss, metrics=METRICS)

    # For testing optimizers
    if(optimizer == 'Adam'):
        model.compile(optimizer=Adam(learning_rate=scheduler), loss=composite_loss, metrics=METRICS)
    elif(optimizer == 'AdamW'):
        model.compile(optimizer=AdamW(learning_rate=scheduler), loss=composite_loss, metrics=METRICS)
    # For testing SGD momentum values
    elif(optimizer[:3] == 'SGD'):
        # pass optimizer in form "SGD - momentum"
        mom_str = optimizer[6:]  # 6 to account for 'SGD - '
        mom = float(mom_str)
        model.compile(optimizer=SGD(learning_rate=scheduler, momentum=mom), loss=composite_loss, metrics=METRICS)
    else:
        print("### Using Default Adam Optimizer ###")
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss=composite_loss, metrics=METRICS)

    
    ### Original is on top
    pretrained_weights = None
    # weight_directory = './weights'
    # pretrained_weights = os.path.join(weight_directory,'model_r2udensenet.hdf5')

    if(pretrained_weights):
        model.load_weights(pretrained_weights)
        
    return model

# to show summary of model
# r2udensenet()