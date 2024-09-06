### Imports
from scipy import io as sio
from r2udensenet_1d import r2udensenet
#from dense_unet import denseunet
# from unet import unet_model
# from res_unet import resunet
# from new_r2unet import r2unet
from data2D_ucsf_1d import load_train_data, load_test_data
import os
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from keras.models import Model
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imsave
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score, precision_recall_curve
from keras.losses import BinaryCrossentropy
from sklearn.preprocessing import binarize
import pdb
from tqdm import tqdm

### For custom dice & IoU functions
from tensorflow.keras import backend as K

### To enable GPU usage
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


### Metrics for testing model
def IoU_calc(y_true, y_pred):
    smooth = 0.00001
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

def dice_coef(y_true, y_pred):
    smooth = 0.00001
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    print(f'y_true_f: {y_true_f}')
    print(f'y_pred_f: {y_pred_f}')
    print(f'intersection: {intersection}')
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


#################### PREDICTION OF THE MODEL WITH MONTE CARLO DROPOUT ####################
def predict(weight_directory, weight_file, pred_directory, saveImgs = False, opt_thresh = 0.5):
    print('========== Beginning of Prediction With Monte Carlo Dropout ==========')

    # Perform normalization in data loader so that the pixel range in each image is 0 - 1
    images_test, mask_test = load_test_data()
    images_test = images_test.astype('float32')


    images_test_T2 = images_test[:,:,:,0]
    images_test_mean = np.mean(images_test_T2)
    images_test_std = np.std(images_test_T2)
    images_test_T2 = (images_test_T2 - images_test_mean)/images_test_std
    # Shape: (n, height, width, 1)
    images_test_T2 = np.expand_dims(images_test_T2, axis = 3)
    images_test = images_test_T2

    # Ground truth masks
    masks_gt = mask_test

    # Load the model and trained weights
    model = r2udensenet()
    model.load_weights(os.path.join(weight_directory, weight_file))

    # Monte Carlo Dropout Predictions
    mcd_iterations = 50
    image_blank = [] 

    # tqdm is used to display a progress bar
    for _ in tqdm(range(mcd_iterations), desc="Monte Carlo Dropout Inference"):
        single_masks_pred = model(images_test, training=True)  # Enable dropout during inference
        single_masks_pred = np.squeeze(single_masks_pred, axis=3)  # Squeeze extra dimension
        image_blank.append(single_masks_pred)

    # Convert list to array containing every dropout prediction
    image_tensor = np.stack(image_blank, axis=0)  # Shape: (mcd_iterations, n, height, width)

    # Compute the mean over multiple dropout predictions
    final_masks_pred = np.mean(image_tensor, axis=0)  # Shape: (n, height, width)
    print(f'Final output shape: {final_masks_pred.shape}')

    # OPTIONAL - try with and without
    # Clamp output between 0 and 1
    final_masks_pred = np.clip(final_masks_pred, 0.0, 1.0)

    # Calculate pixel-wise standard deviation (uncertainty estimate)
    variance = np.var(image_tensor, axis=0)
    standard_deviation = np.sqrt(variance)

    # Initialize a list to store standard deviations for each image
    std_devs_per_image = []

    for i in range(standard_deviation.shape[0]):
        single_image_stds = standard_deviation[i, :, :]

        # Compute the average standard deviation for the current image
        avg_image_std = np.mean(single_image_stds)
        
        std_devs_per_image.append(avg_image_std)

    print(f'Standard deviations for each image: {std_devs_per_image}')

    # Calculating metrics and saving images
    dice = []
    precision = []
    recall = []
    auc = []
    IoU = []
    weighted_binaryCE = []
    dice_actual = []

    print('Threshold', opt_thresh)
    print(f'np.unique(final_masks_pred): {np.unique(final_masks_pred, return_counts=True)}')
    print(f'np.unique(masks_gt): {np.unique(masks_gt, return_counts=True)}')

    final_masks_pred[final_masks_pred >= opt_thresh] = 1
    final_masks_pred[final_masks_pred < opt_thresh] = 0
    final_masks_pred_save = (final_masks_pred*255.).astype(np.uint8)

    print(f'np.unique(final_masks_pred): {np.unique(final_masks_pred, return_counts=True)}')
    print(f'np.unique(masks_gt): {np.unique(masks_gt, return_counts=True)}')
    
    for i in range(0, final_masks_pred.shape[0]): # final_masks_pred.shape[0]
        if(saveImgs):
            imsave(os.path.join(pred_directory,  str(i) + '_pred' + '.png' ), final_masks_pred_save[i])
            # img_save = (images_test_T2[i]).astype(np.uint8)
            imsave(os.path.join(pred_directory,  str(i) + '_img' + '.png' ), images_test_T2[i])
            # gt_save = (gt*255).astype(np.uint8)
            imsave(os.path.join(pred_directory,  str(i) + '_gt' + '.png' ), masks_gt[i])

        gt = masks_gt[i,:,:]
        test = final_masks_pred[i,:,:]

        #
        print(f'np.unique(gt): {np.unique(gt, return_counts=True)}')
        print(f'np.unique(test): {np.unique(test, return_counts=True)}')
        dice_actual1 = dice_coef(gt.astype('float32'), test)
        print(f'dice_actual1: {dice_actual1}')
        dice_actual.append(dice_actual1)
        #

        # gt = binarize(gt, threshold = 0.5)
        
        dice1 = f1_score(gt.flatten(), test.flatten(), average = 'binary')
        dice.append(dice1)
        # print(f'img {i} dice: {dice1}')
        # if dice1 < 0.7 or dice1 > 0.95:
        #     print(f'img {i} dice: {dice1}')

        precision1 = precision_score(gt.flatten(), test.flatten(), average = 'binary')
        precision.append(precision1)
        
        recall1 = recall_score(gt.flatten(), test.flatten(), average = 'binary')
        recall.append(recall1)
        
        try:
            auc1 = roc_auc_score(gt.flatten(), test.flatten())
        except ValueError:
            auc1 = 0
        # auc1 = roc_auc_score(gt.flatten(), test.flatten())
        auc.append(auc1)

        gt_temp = gt.astype('float32')
        
        # print(f'np.unique(gt): {np.unique(gt)}')
        # print(f'np.unique(test): {np.unique(test)}')
        # print(f'np.unique(gt_temp): {np.unique(gt_temp)}')
        # print(f'gt.shape: {gt.shape}')
        # print(f'test.shape: {test.shape}')
        # print(f'gt_temp.shape: {gt_temp.shape}')

        IoU1 = IoU_calc(gt_temp, test)
        IoU.append(IoU1)

        weighted_binaryCE1 = weighted_bce(gt_temp, test)
        weighted_binaryCE.append(weighted_binaryCE1)

        print(f'### img {i} ###')
        print(f'dice1: {dice1}')
        print(f'precision1: {precision1}')
        print(f'recall1: {recall1}')
        print(f'auc1: {auc1}')
        print(f'IoU1: {IoU1}')
        print(f'weighted_binaryCE1: {weighted_binaryCE1}')

    dice_mean = np.mean(dice)
    dice_std = np.std(dice)
    
    prec_mean = np.mean(precision)
    prec_std = np.std(precision)
    
    recall_mean = np.mean(recall)
    recall_std = np.std(recall)
    
    auc_mean = np.mean(auc)
    auc_std = np.std(auc)

    iou_mean = np.mean(IoU)
    iou_std = np.std(IoU)

    bce_mean = np.mean(weighted_binaryCE1)
    bce_std = np.std(weighted_binaryCE1)

    
    print(f'Dice Mean = {dice_mean}')
    print(f'Standard Dev Dice = {dice_std}')
    
    print(f'Precision Mean = {prec_mean}')
    print(f'Standard Dev Precision = {prec_std}')
    
    print(f'Recall Mean = {recall_mean}')
    print(f'Standard Dev Recall = {recall_std}')
    
    print(f'AUC Mean = {auc_mean}')
    print(f'Standard Dev AUC = {auc_std}')

    print(f'IoU Mean = {iou_mean}')
    print(f'Standard Dev IoU = {iou_std}')

    print(f'BCE Mean = {bce_mean}')
    print(f'Standard Dev BCE = {bce_std}')

    print(f'Actual Dice Mean = {np.mean(dice_actual)}')
    print(f'Standard Dev Actual Dice = {np.std(dice_actual)}')

    print('========== End of Prediction ==========')



if __name__ == '__main__':
    date = '8_19_23'
    trial_number = 3
    model_number = 3
    saveImgs = False

    weight_directory = './weights_2023/weights_' + date + '/trial_' + str(trial_number)
    weight_file = 'model_r2udensenet' + str(model_number) + '.hdf5'
    pred_directory = './output/r2udensenet/' + date + '/trial_' + str(trial_number) + '/model_' + str(model_number)
    
    print(f'weight_directory: {weight_directory}')
    print(f'weight_file: {weight_file}')
    print(f'pred_directory: {pred_directory}')

    if not os.path.exists(pred_directory):
        print("DOESN'T EXIST")
        os.makedirs(pred_directory)
        print(f'Directory created: {pred_directory}')

    opt_thresh = 0.5
    dice = predict(weight_directory, weight_file, pred_directory, saveImgs, opt_thresh)
