import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
# Set TensorFlow logging level to ERROR
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from scipy import io as sio
from r2udensenet_1d import r2udensenet

# from dense_unet import denseunet
# from unet import unet_model
# from res_unet import resunet
# from new_r2unet import r2unet

# from data2D_single import load_train_data, load_test_data
# from data2D_merge import load_train_data, load_test_data
from data2D import load_train_data, load_test_data
# from data_multiple import load_train_data, load_test_data

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from keras.models import Model
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
import warnings
# import cv2

### for custom dice & IoU functions
from tensorflow.keras import backend as K

warnings.filterwarnings("ignore")

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Clear any previous models and session
tf.keras.backend.clear_session()

def gen_precision_recall(weight_directory, weight_file):
    print('====== Generate Precision Recall Curve ========')
    images_train, mask_train = load_train_data()

    print('### 1 ###')
    print(f'mask_train.shape: {mask_train.shape}')
    print(f'images_train.shape: {images_train.shape}')

    images_train = images_train.astype('float32')
    mask_train = mask_train.astype('float32')

    print('### 2 ###')
    print(f'mask_train.shape: {mask_train.shape}')
    print(f'images_train.shape: {images_train.shape}')

    images_train_mean = np.mean(images_train)
    images_train_std = np.std(images_train)
    images_train = (images_train - images_train_mean)/images_train_std

    mask_train /= 255.

    print('### 3 ###')
    print(f'mask_train.shape: {mask_train.shape}')
    print(f'images_train.shape: {images_train.shape}')
    
    mask_train = mask_train.ravel()
    print('mask_train.shape after ravel', mask_train.shape)
    mask_train = mask_train.reshape(-1,1)
    print('mask_train.shape after reshape', mask_train.shape)
    mask_train = binarize(mask_train, threshold = 0.5)
    print('mask_train.shape final', mask_train.shape)

    #####
    # images_train_final = np.append(images_train, images_train_T1, axis = 3)
    images_train_final = images_train
    # pdb.set_trace()

    print(f'images_train_final.shape: {images_train_final.shape}')
    print(f'images_train_final.shape: {images_train_final.shape}')

    model = r2udensenet()
    # pdb.set_trace()

    # weight_directory = 'weights'
    # model.load_weights(os.path.join(weight_directory,'model_r2udensenet.hdf5'))
    model.load_weights(os.path.join(weight_directory, weight_file))
    y_pred = model.predict(images_train_final, batch_size = 1, verbose = 1)  
    y_pred = np.squeeze(y_pred, axis = 3)
    
    y_test = mask_train
    nn_fpr_keras, nn_tpr_keras, nn_thresholds_keras = precision_recall_curve(y_test.ravel(), y_pred.ravel())
    diff = np.abs(nn_fpr_keras-nn_tpr_keras)
    place = np.argmin(diff)
    print('Optimal Threshold = ', str(nn_thresholds_keras[place]))
    print('Recall = ',nn_fpr_keras[place])
    print('Precision = ',nn_tpr_keras[place])
    plt.plot(nn_tpr_keras, nn_fpr_keras)
    plt.ylabel('Precision',fontweight='bold',fontsize = 20)
    plt.xlabel('Recall',fontweight='bold',fontsize = 20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.savefig(f'{pred_directory}/precision_recall.png')
    opt_thresh = nn_thresholds_keras[place]
    #
    print(f'diff: {len(diff)}')
    print(f'nn_thresholds_keras len: {len(nn_thresholds_keras)}')
    print(f'place: {place}')
    # for i in diff:
    #     print(f'diff: {i}')
    #
    return opt_thresh
    
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
    # print(f'y_true_f: {y_true_f}')
    # print(f'y_pred_f: {y_pred_f}')
    # print(f'intersection: {intersection}')
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    

################################### PREDICTION OF THE NETWORK ###################################    
def predict(weight_directory, weight_file, dataset_path, pred_directory, saveImgs = False, printInfo = False, opt_thresh = 0.5):
    print('============= Beginning of Prediction ================')
    # images_test, mask_test = load_test_data()
    images_test, mask_test = load_test_data(dataset_path)
    images_test = images_test.astype('float32')
    
    images_test_T2 = images_test[:,:,:,0]
    # Normalization

    # images_test_mean = np.mean(images_test_T2)
    # images_test_std = np.std(images_test_T2)
    # images_test_T2 = (images_test_T2 - images_test_mean)/images_test_std

    for i in range(images_test_T2.shape[0]):
        sl = images_test_T2[i]

        sl_min = np.min(sl)
        sl_max = np.max(sl)

        sl_normalized = (sl - sl_min) / (sl_max - sl_min)
        images_test_T2[i] = sl_normalized

    images_test_T2 = np.expand_dims(images_test_T2, axis = 3)

    # images_test_T2 = images_test_T2[117:118]
    # mask_test = mask_test[117:118]

    images_test = images_test_T2
    masks_gt = mask_test

    # print(images_test.shape)
    # print(mask_test.shape)

    model = r2udensenet()

    model.load_weights(os.path.join(weight_directory, weight_file))
    masks_pred = model.predict(images_test, batch_size=1, verbose=1)
    # print(f'#1 masks_pred.shape: {masks_pred.shape}')
    # print(f'#1 len(np.unique(masks_pred)): {len(np.unique(masks_pred))}')
    # print(f'#1 np.unique(masks_pred): {np.unique(masks_pred)[0]} - {np.unique(masks_pred)[-1]}')
    masks_pred = np.squeeze(masks_pred, axis = 3)
    # print(f'#2 masks_pred.shape: {masks_pred.shape}')
    # print(f'#2 len(np.unique(masks_pred)): {len(np.unique(masks_pred))}')
    # print(f'#2 np.unique(masks_pred): {np.unique(masks_pred)[0]} - {np.unique(masks_pred)[-1]}')
    
    ### moved to main function
    if not os.path.exists(pred_directory):
        # os.mkdir(pred_directory)
        os.makedirs(pred_directory)
    
    dice = []
    precision = []
    recall = []
    auc = []
    IoU = []
    weighted_binaryCE = []
    dice_actual = []
    
    # print('Threshold', opt_thresh)
    # print(f'np.unique(masks_pred): {np.unique(masks_pred, return_counts=True)}')
    # print(f'np.unique(masks_gt): {np.unique(masks_gt, return_counts=True)}')

    masks_pred[masks_pred >= opt_thresh] = 1
    masks_pred[masks_pred < opt_thresh] = 0
    masks_pred_save = (masks_pred*255.).astype(np.uint8)
    
    # print(f'np.unique(masks_pred): {np.unique(masks_pred, return_counts=True)}')
    # print(f'np.unique(masks_gt): {np.unique(masks_gt, return_counts=True)}')
    
    for i in range(0, masks_pred.shape[0]): # masks_pred.shape[0]
        if(saveImgs):
            plt.imshow(images_test_T2[i], cmap='gray')
            plt.imshow(masks_pred_save[i], alpha=0.4)
            plt.savefig(os.path.join(pred_directory,  str(i) + '_pred_overlay.png'))
            plt.close()

            plt.imshow(images_test_T2[i], cmap='gray')
            plt.imshow(masks_gt[i], alpha=0.4)
            plt.savefig(os.path.join(pred_directory,  str(i) + '_gt_overlay.png'))
            plt.close()

            imsave(os.path.join(pred_directory,  str(i) + '_pred' + '.png' ), masks_pred_save[i])
            # img_save = (images_test_T2[i]).astype(np.uint8)
            imsave(os.path.join(pred_directory,  str(i) + '_img' + '.png' ), images_test_T2[i])
            # gt_save = (gt*255).astype(np.uint8)
            imsave(os.path.join(pred_directory,  str(i) + '_gt' + '.png' ), masks_gt[i])

            plt.imshow(images_test_T2[i], cmap='gray')
            plt.imshow(masks_pred_save[i], alpha=0.4)
            plt.savefig(os.path.join(pred_directory,  str(i) + '_overlay.png'))
            plt.close()

            # mask1=masks_pred_save[i]
            # mask2=masks_gt[i]
            # result_image_1 = (mask1 / 1) * (1 - mask2 / 1) * 1
            # # result_image_1 = cv2.bitwise_and(masks_pred_save[i], cv2.bitwise_not(masks_gt[i]))
            # imsave(os.path.join(pred_directory,  str(i) + '_pred_mask' + '.png' ), result_image_1)

            # result_image_2 = (mask2 / 1) * (1 - mask1 / 1) * 1
            # # result_image_2 = cv2.bitwise_and(masks_gt[i], cv2.bitwise_not(masks_pred_save[i]))
            # imsave(os.path.join(pred_directory,  str(i) + '_gt_pred' + '.png' ), result_image_2)

            pred_gt = np.zeros((192, 192))
            # for iy, ix in np.ndindex(pred_gt.shape):
            for x in range(0, 192):
                for y in range(0, 192):
                    # print(f'masks_pred_save[i, x, y]: {masks_pred_save[i, x, y]}')
                    # print(f'masks_gt[i, x, y]: {masks_gt[i, x, y]}')
                    if masks_pred_save[i, x, y] == 1 and masks_gt[i, x, y] == 0:
                        pred_gt[x, y] = 1
                    else:
                        pred_gt[x, y] = 0
            print(f'np.unique(pred_gt, return_counts=True): {np.unique(pred_gt, return_counts=True)}')
            plt.imshow(images_test_T2[i], cmap='gray')
            plt.imshow(pred_gt, alpha=0.4)
            plt.savefig(os.path.join(pred_directory,  str(i) + '_pred_mask.png'))
            plt.close()

            gt_pred = np.zeros((192, 192))
            # for iy, ix in np.ndindex(pred_gt.shape):
            for x in range(0, 192):
                for y in range(0, 192):
                    if masks_gt[i, x, y] == 1 and masks_pred_save[i, x, y] == 0:
                        gt_pred[x, y] = 1
                    else:
                        gt_pred[x, y] = 0
            print(f'np.unique(pred_gt, return_counts=True): {np.unique(gt_pred, return_counts=True)}')
            plt.imshow(images_test_T2[i], cmap='gray')
            plt.imshow(gt_pred, alpha=0.4)
            plt.savefig(os.path.join(pred_directory,  str(i) + '_gt_pred.png'))
            plt.close()

        gt = masks_gt[i,:,:]
        test = masks_pred[i,:,:]

        #
        if(printInfo):
            print(f'np.unique(gt): {np.unique(gt, return_counts=True)}')
            print(f'np.unique(test): {np.unique(test, return_counts=True)}')
        dice_actual1 = dice_coef(gt.astype('float32'), test)
        if(printInfo):
            print(f'dice_actual1: {dice_actual1}')
        dice_actual.append(dice_actual1)
        #

        # gt = binarize(gt, threshold = 0.5)
        
        dice1 = f1_score(gt.flatten(), test.flatten(), average = 'binary', zero_division = 1.0)
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

        if(printInfo):
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
    print(f'dice: {dice}')
    
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

    print('============= End of Prediction ================')

    print(f'weight_directory: {weight_directory}')
    print(f'weight_file: {weight_file}')
    
    return dice


'''if __name__ == '__main__':
    date = '07_22_24'
    trial_number = 1
    model_number = 1
    saveImgs = False
    printInfo = True

    weight_directory = './weights_2024/weights_' + date + '/trial_' + str(trial_number)
    weight_file = 'model_r2udensenet' + str(model_number) + '.hdf5'
    # pred_directory = './output/r2udensenet/' + date + '/trial_' + str(trial_number) + '/model_' + str(model_number)
    pred_directory = './output/debug/07_13_24'
    
    print(f'weight_directory: {weight_directory}')
    print(f'weight_file: {weight_file}')
    print(f'pred_directory: {pred_directory}')

    if not os.path.exists(pred_directory):
        print("DOESN'T EXIST")
        os.makedirs(pred_directory)
        print(f'Directory created: {pred_directory}')

    # opt_thresh = gen_precision_recall(weight_directory, weight_file)
    # print(f'opt_thresh: {opt_thresh}')

    opt_thresh = 0.5

    dice = predict(weight_directory, weight_file, pred_directory, saveImgs, printInfo, opt_thresh)'''

'''if __name__ == '__main__':

    date = '8_19_23'
    trial_number = 4
    saveImgs = False
    printInfo = False
    weight_directory = './weights_2023/weights_' + date + '/trial_' + str(trial_number)

    for i in range(1, 6):
        model_number = i
        print(f'############################## Trial #{model_number} ##############################')

        weight_file = 'model_r2udensenet' + str(model_number) + '.hdf5'
        # pred_directory = './output/r2udensenet/' + date + '/trial_' + str(trial_number) + '/model_' + str(model_number)
        
        print(f'weight_directory: {weight_directory}')
        print(f'weight_file: {weight_file}')

        # print(f'pred_directory: {pred_directory}')

        # if not os.path.exists(pred_directory):
        #     print("DOESN'T EXIST")
        #     os.makedirs(pred_directory)
        #     print(f'Directory created: {pred_directory}')

        opt_thresh = 0.5

        dice = predict(weight_directory, weight_file, pred_directory, saveImgs, printInfo, opt_thresh)'''

if __name__ == '__main__':

    date = '08_24_24' #'08_07_24' # '08_04_24' # '07_30_24'
    model_number = 1
    saveImgs = False
    printInfo = False
    # dataset_path = ['/data/satvik/data/liver/test']
    dataset_path = ['/data/satvik/data/tibia/test']
    # dataset_path = ['/data/satvik/data_2023/new_data_110623/test', '/data/satvik/data/liver/test', '/data/satvik/data/tibia/test']
    # dataset_path = '/data/satvik/data_2023/new_data_110623/test'
    # test_trials = [27, 34, 41, 30, 26]
    # test_trials = [17, 33, 1, 73, 53, 18, 34, 54]
    test_trials = [1, 2, 3, 4, 5, 6]
    # test_trials = [3]

    for i in test_trials:
        trial_number = i
        # weight_directory = './weights_2024/weights_' + date + '/trial_' + str(trial_number)
        weight_directory = './weights_2024/weights_' + date + '/trial_' + str(trial_number)

        print(f'############################## Trial #{trial_number} ##############################')

        weight_file = 'model_r2udensenet' + str(model_number) + '.hdf5'
        pred_directory = './output/r2udensenet/' + date + '/trial_' + str(trial_number) + '/model_' + str(model_number)
        # pred_directory = './output/debug/07_13_24'
        
        print(f'weight_directory: {weight_directory}')
        print(f'weight_file: {weight_file}')
        print(f'dataset_path: {dataset_path}')

        # print(f'pred_directory: {pred_directory}')

        # if not os.path.exists(pred_directory):
        #     print("DOESN'T EXIST")
        #     os.makedirs(pred_directory)
        #     print(f'Directory created: {pred_directory}')

        opt_thresh = 0.5

        dice = predict(weight_directory, weight_file, dataset_path, pred_directory, saveImgs, printInfo, opt_thresh)