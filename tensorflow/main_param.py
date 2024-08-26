# Set up GPU before importing tensorflow
import os
print('############ 1 ############')
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# https://stackoverflow.com/questions/53698035/failed-to-get-convolution-algorithm-this-is-probably-because-cudnn-failed-to-in
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import tensorflow as tf
print('############ 2 ############')
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("Current GPU:")
print(tf.config.list_physical_devices('GPU'))
# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Imports
import numpy as np
import matplotlib.pyplot as plt
import pdb
import warnings
import csv

from time import time
from datetime import datetime
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import CSVLogger
# from keras.callbacks import Callback
from tensorflow.keras.callbacks import Callback
# from keras.models import Model
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import binarize
from sklearn.utils import class_weight
# from sklearn.utils import sample_weight
from sklearn.model_selection import KFold

# Importing external functions
from r2udensenet_copy import r2udensenet
'''from dense_unet import denseunet
from unet import unet_model
from res_unet import resunet
from new_r2unet import r2unet'''
# data2D_single, data2D_merge, data2D_original
# from data2D_merge import load_train_data, load_test_data
from data2D import load_train_data, load_test_data

# TO REMOVE
# from scipy import io as sio
# from skimage.io import imsave

warnings.filterwarnings("ignore")

# def gen_precision_recall():
# Function not coppied from main_finetune.pf

#################### Training of the Network by Five Fold Cross Validation #################################

def train(train_params):
    composite_loss = []
    dice_loss = []
    dice = []
    precision = []
    recall = []
    auc = []
    accuracy = []
    IoU_test = []
    binaryCE_original = [] # without equal weighting of background and tumor
    bce = []

    date = train_params['Date']
    trial_number = train_params['Trial Number']

    weight_save_directory, log_directory = return_directories(date, trial_number)

    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # current_time = time.localtime()
    # run_time = time.strftime("%Y-%m-%d %H:%M:%S", current_time)
    info_filename = os.path.join(log_directory, 'info.txt')
    create_info_file(run_time, train_params, info_filename)

    kfold = KFold(n_splits = 5, shuffle = True, random_state = 1)

    # batch size
    batch_s = train_params['Batch Size']

    # bool_load_weights
    bool_load_weights = train_params['Load Weights']

    # pdb.set_trace()
    print(f'kfold: {kfold}')

    dataset_path = train_params['Dataset']
    images_train, mask_train = load_train_data(dataset_path)

    images_train = images_train.astype('float64')
    mask_train = mask_train.astype('float64')

    # Normalization - inputs between 0 and 1
    # images_train /= 255

    # images_train_mean = np.mean(images_train)
    # images_train_std = np.std(images_train)
    # images_train = (images_train - images_train_mean)/images_train_std

    for i in range(images_train.shape[0]):
        sl = images_train[i]

        sl_min = np.min(sl)
        sl_max = np.max(sl)

        sl_normalized = (sl - sl_min) / (sl_max - sl_min)
        images_train[i] = sl_normalized
    
    # Converts masks from 0/255 to 0/1
    # Performed in data loader
    # if(np.max(mask_train) == 255):
    #     print("Rescaling masks from 0 - 255 to 0 - 1")
    #     mask_train /= 255.

    # Deprecated - Used for experimenting with class_weights
    # mask_train_temp = np.ravel(mask_train)
    # class_weights = class_weight.compute_class_weight('balanced',
    #                                                   np.unique(mask_train_temp),
    #                                                   mask_train_temp)
    # class_weights_dict = dict(zip(np.unique(mask_train_temp), class_weights))
    # print(f'class_weights_dict: {class_weights_dict}')
    # pdb.set_trace()

    ## Look into changing values
    # proability if the augmentations should happen
    # change frequency or magnitude

    # datagen_train = ImageDataGenerator(
    #                 rotation_range=0.2,
    #                 width_shift_range=0.05,
    #                 height_shift_range=0.05,
    #                 shear_range=0.05,
    #                 zoom_range=0.05,
    #                 horizontal_flip=True,
    #                 vertical_flip=True,
    #                 fill_mode='nearest'
    #                 )
    
    datagen_no_augmentation = ImageDataGenerator(
                    # rotation_range=0.2,
                    # width_shift_range=0.05,
                    # height_shift_range=0.05,
                    # shear_range=0.05,
                    # zoom_range=0.05,
                    # horizontal_flip=True,
                    # vertical_flip=True,
                    # fill_mode='nearest'
                    )
    
    #
    datagen_augmentation = ImageDataGenerator(
                    rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    # horizontal_flip=True,
                    # vertical_flip=True,
                    fill_mode='nearest'
                    )

    augmentation_prob = train_params['Augmentation Probability']
    # augmentation_prob = 1

    def generate_generator(generator, images_train, mask_train, batch_size, bool_shuffle=True):
        # modified
        train_gen = generator.flow(images_train, y=None, batch_size = batch_size, shuffle = bool_shuffle, seed = 100, 
                                              sample_weight = None)
    
        label_gen = generator.flow(mask_train, y=None, batch_size = batch_size, shuffle = bool_shuffle, seed = 100, 
                                              sample_weight = None)
        
        while True:
            image_T2 = train_gen.next()
            mask = label_gen.next()
            # Added
            if np.unique(mask).shape[0] > 2:
                mask[mask >= 0.5] = 1.
                mask[mask < 0.5] = 0.
            #
            yield image_T2,mask

    # Old
    '''def custom_generator(generator, augmentation_probability):
        for x_batch, y_batch in generator:
            for i in range(len(x_batch)):
                if np.random.random() < augmentation_probability:
                    x_batch[i] = datagen_augmentation.random_transform(x_batch[i])

            yield x_batch, y_batch'''

    # Modified
    def custom_generator(no_aug_generator, augmentation_generator, images_train, mask_train, augmentation_probability, batch_size, bool_shuffle=True):
        base_generator = generate_generator(no_aug_generator, images_train, mask_train, batch_size, bool_shuffle)
        
        while True:
            x_batch, y_batch = next(base_generator)
            
            # Debug statement to print shapes of batches
            # print(f"x_batch shape: {x_batch.shape}, y_batch shape: {y_batch.shape}")

            current_batch_size = x_batch.shape[0]
            
            for i in range(current_batch_size):
            # for i in range(batch_size):
                if np.random.random() < augmentation_probability:

                    # Debug
                    # print("Augmenting data")
                    # print(f"x_batch[i]: {x_batch[i].shape}")
                    # print(f"y_batch[i]: {y_batch[i].shape}")

                    # Apply the same random transformation to both x_batch (images) and y_batch (masks)
                    # Concatenate image and mask along the channel axis
                    combined = np.concatenate([x_batch[i], y_batch[i]], axis=-1)

                    # Debug statement to check shapes before transformation
                    # print(f"combined shape before transform: {combined.shape}")

                    # Apply random transformation
                    combined = augmentation_generator.random_transform(combined)
                
                    # Debug statement to check shapes after transformation
                    # print(f"combined shape after transform: {combined.shape}")
                

                    # Split back into image and mask
                    x_batch[i] = combined[..., :-1]
                    y_batch[i] = combined[..., -1][..., np.newaxis]

                    # debug
                    # print(f"x_batch[i]: {x_batch[i].shape}")
                    # print(f"y_batch[i]: {y_batch[i].shape}")
                    
            yield x_batch, y_batch

    # train_generator = generate_generator(image_datagen, mask_datagen, images_train, mask_train)

    # pdb.set_trace()
    # print(f'kfold.split(images_train): {kfold.split(images_train)}')

    fold_no = 1
    for train_index, validation_index in kfold.split(images_train):

        # pdb.set_trace()
        # print(f'train_index: {train_index}')
        print(f'train_index: {train_index[0]} - {train_index[-1]}')
        # print(f'train_index.shape: {train_index.shape}')
        # print(f'train_index.dtype: {train_index.dtype}')
        # print(f'validation_index: {validation_index}')
        print(f'validation_index: {validation_index[0]} - {validation_index[-1]}')
        # print(f'validation_index.shape: {validation_index.shape}')
        # print(f'validation_index.dtype: {validation_index.dtype}')

        trainData_T2 = images_train[train_index]
        trainMask = mask_train[train_index]
    
        validationData_T2 = images_train[validation_index]
        validationMask = mask_train[validation_index]
    
        # train_generator = generate_generator(datagen_train, trainData_T2, trainMask)
        # train_generator_base = generate_generator(datagen_no_augmentation, trainData_T2, trainMask, batch_s)
        train_generator = custom_generator(datagen_no_augmentation, datagen_augmentation, trainData_T2, trainMask, augmentation_prob, batch_s)
        validation_generator = generate_generator(datagen_no_augmentation, validationData_T2, validationMask, batch_s, False)

        # model = r2udensenet()
        opt = train_params['Optimizer']
        lr = train_params['Learning Rate']
        sch = train_params['Scheduler']
        eps = train_params['Epochs']

        # Calculating decay steps to decay learning rate for schedulers
        num_decays = 10 # number of times the learning rate gets decayed
        total_samples = 1932 # total number of steps with batch size of 1
        total_samples = len(train_index)
        steps_per_epoch = total_samples / batch_s
        total_steps = steps_per_epoch * eps
        ds = int(total_steps / num_decays)

        # When using linear scheduler
        # Coontinuously decays learning rate continuously so pass total_steps as ds
        if(sch == 'Linear' or sch == 'CosineDecay'):
            ds = total_steps

        print(f"opt: {opt}")
        print(f"lr: {lr}")
        print(f"sch: {sch}")
        print(f"ds: {ds}")
        model = r2udensenet(opt, lr, sch, ds)

        # load weights
        # unlock model

        if bool_load_weights:
            load_date = '8_11_23'
            load_trial_number = '3'
            model_number = '2'
            weight_load_directory = './weights_2023/weights_' + load_date + '/trial_' + str(load_trial_number)
            model_name = 'model_r2udensenet' + str(model_number) + '.hdf5'
        
            model.load_weights(os.path.join(weight_load_directory, model_name))
            model.trainable = True
            print(f'########## Weights Loaded from {model_name} ##########')
            print("weights:", len(model.weights))
            print("trainable_weights:", len(model.trainable_weights))
            print("non_trainable_weights:", len(model.non_trainable_weights))
        else:
            print('########## NO Weights Loaded ##########')

        # print('########## Summary ##########')
        # model.summary()
        # for debugging
        # pdb.set_trace()


        model_checkpoint = ModelCheckpoint(
            os.path.join(weight_save_directory, get_model_name(fold_no)),
            # monitor = 'loss',
            # monitor = 'val_loss',
            monitor = train_params['Monitor Param'],
            verbose = 1,
            mode = 'min', # min because monitoring loss
            save_best_only = True)
    
        logger = CSVLogger(
            os.path.join(log_directory, get_log_name(fold_no)),
            separator = ',',
            append = False)

        # pdb.set_trace()
        # print('##### Before model training #####')

        start = time()
        history = model.fit_generator(
            train_generator,
            steps_per_epoch = len(trainData_T2)/batch_s, # '/2' = bactch size
            epochs = train_params['Epochs'],
            verbose = 1,
            validation_data = validation_generator,
            validation_steps = len(validationData_T2)/batch_s, # 26 should be num validation data / batchsize
            # class_weight = class_weights_dict, #class_weights_dict, {0.0: 0.5, 1.0: 10}
            # sample_weight = class_weights_dict,
            callbacks = [model_checkpoint, logger])
        print("===== K Fold Validation Step ======", fold_no)
        
        plot_directory = log_directory + '/model' + str(fold_no)
        if not os.path.exists(plot_directory):
            os.makedirs(plot_directory)

        create_plots(history, fold_no, plot_directory)
        
        
        # val_punch = np.append(validationData_T2,validationData_T1, axis = 3)
        val_punch = validationData_T2
        scores = model.evaluate(val_punch, validationMask, verbose=0)
        composite_loss.append(scores[0])
        dice_loss.append(scores[1])
        dice.append(scores[2])
        precision.append(scores[3])
        recall.append(scores[4])
        auc.append(scores[5])
        accuracy.append(scores[6])
        IoU_test.append(scores[7])
        binaryCE_original.append(scores[8])
        bce.append(scores[9])

        #####
        print(f'val_punch.dtype: {val_punch.dtype}')
        print(f'val_punch.shape: {val_punch.shape}')
        print(f'length of scores: {len(scores)}')
        print(f'composite loss: {composite_loss}')
        print(f'dice_loss: {dice_loss}')
        print(f'dice: {dice}')
        print(f'precision: {precision}')
        print(f'recall: {recall}')
        print(f'auc: {auc}')
        print(f'accuracy: {accuracy}')
        print(f'IoU: {IoU_test}')
        print(f'binaryCE_original: {binaryCE_original}')
        print(f'bce: {bce}')
        print()
        #####

        update_leaderboard(run_time, trial_number, fold_no, dataset_path, opt, lr, batch_s, sch, scores[2], scores[0])

        fold_no = fold_no + 1

        # To only run first kfold
        break

    summary = format_summary(composite_loss, dice_loss, dice, precision, recall, auc, accuracy, IoU_test, binaryCE_original, bce)
    print_summary(summary)
    append_to_info_file(summary, info_filename)

  
def return_directories(date, trial_num):
    weight_save_directory = './weights_2024/weights_' + date + '/trial_' + str(trial_num)
    print(f'weight_save_directory: {weight_save_directory}')
    log_directory = './logs_2024/Training_' + date + '/trial_' + str(trial_num)
    print(f'log_directory: {log_directory}')

    if not os.path.exists(weight_save_directory):
        print("DOESN'T EXIST")
        os.makedirs(weight_save_directory)
        print(f'Directory created: {weight_save_directory}')
    if not os.path.exists(log_directory):
        print("DOESN'T EXIST")
        os.makedirs(log_directory)
        print(f'Directory created: {log_directory}')

    return weight_save_directory, log_directory

def create_plots(history, fold_no, plot_directory):
    plt.title(f'Model #{fold_no}')
    # plt.ylim(0, 1)
    plt.plot(history.history['loss'])  
    plt.plot(history.history['val_loss'])  
    plt.ylabel('Composite Loss(75/25)',fontweight='bold')  
    plt.xlabel('Epochs',fontweight='bold')  
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(f'{plot_directory}/Composite_loss_{fold_no}.png')
    plt.close()
    
    plt.title(f'Model #{fold_no}')
    # plt.ylim(0, 1)
    plt.plot(history.history['dice_coef'])  
    plt.plot(history.history['val_dice_coef'])  
    plt.ylabel('Dice Coefficient',fontweight='bold')  
    plt.xlabel('Epochs',fontweight='bold')  
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(f'{plot_directory}/Dice_coef_{fold_no}.png')
    plt.close()

    ## FIX
    plt.title(f'Model #{fold_no}')
    # plt.ylim(0, 1)
    plt.plot(history.history['dice_loss'])  
    plt.plot(history.history['val_dice_loss'])  
    plt.ylabel('Dice Loss',fontweight='bold')  
    plt.xlabel('Epochs',fontweight='bold')  
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(f'{plot_directory}/Dice_loss_{fold_no}.png')
    plt.close()

    plt.title(f'Model #{fold_no}')
    # plt.ylim(0, 1)
    plt.plot(history.history['weighted_bce'])  
    plt.plot(history.history['val_weighted_bce'])  
    plt.ylabel('Weighted Binary Cross-Entropy Loss',fontweight='bold')  
    plt.xlabel('Epochs',fontweight='bold')  
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(f'{plot_directory}/BCE_loss_{fold_no}.png')
    plt.close()

    plt.title(f'Model #{fold_no}')
    # plt.ylim(0, 1)
    plt.plot(history.history['IoU'])  
    plt.plot(history.history['val_IoU'])  
    plt.ylabel('IoU',fontweight='bold')
    plt.xlabel('Epochs',fontweight='bold')  
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(f'{plot_directory}/IoU_{fold_no}.png')
    plt.close()

    print(f'Plots saved as in {plot_directory}')

def format_summary(composite_loss, dice_loss, dice, precision, recall, auc, accuracy, IoU_test, binaryCE_original, bce):
    summary_lines = [
        f'Composite Loss == {composite_loss} Mean Composite Loss == {np.mean(composite_loss)}',
        f'Dice Loss == {dice_loss} Mean Dice Loss == {np.mean(dice_loss)}',
        f'Dice Coefficient == {dice} Mean Dice Coefficient == {np.mean(dice)}',
        f'Precision == {precision} Mean Precision == {np.mean(precision)}',
        f'Recall == {recall} Mean Recall == {np.mean(recall)}',
        f'AUC == {auc} Mean AUC == {np.mean(auc)}',
        f'Accuracy == {accuracy} Mean Accuracy == {np.mean(accuracy)}',
        f'IoU == {IoU_test} Mean IoU == {np.mean(IoU_test)}',
        f'BCE original (equal weighting) == {binaryCE_original} Mean BCE original == {np.mean(binaryCE_original)}',
        f'BCE (unweighted) == {bce} Mean BCE == {np.mean(bce)}',
    ]

    return summary_lines

def print_summary(lines):
    for line in lines:
        print(line)

def get_model_name(k):
    return 'model_r2udensenet'+str(k)+'.hdf5'
    
def get_log_name(k):
    return 'log_r2udensenet'+str(k)+'.csv'

def create_info_file(date, info_dict, filename='info.txt'):
    # Gather information
    model_name = "r2udensenet"
    # dataset_name = "/data/satvik/data_2023/new_data_110623"

    # Create the info string
    info_lines = [
        f"Date: {date}",
        f"Model Name: {model_name}",
        # f"Dataset Name: {dataset_name}",
        f"Additional Information: Testing Optimizer"
    ]
    for key, value in info_dict.items():
        info_lines.append(f"{key}: {value}")

    # Write the info to a file
    with open(filename, 'w') as file:
        for line in info_lines:
            file.write(line + '\n')
        file.write('\n')

    print(f"Create info.txt file at {filename}")

def append_to_info_file(lines, filename='info.txt'):
    with open(filename, 'a') as file:
        for line in lines:
            file.write(line + '\n')

def update_leaderboard(date, trial_number, kfold_no, dataset_list, optimizer, learning_rate, batch_size, scheduler, val_dice, val_loss):
    model_name = get_model_name(kfold_no)
    dataset = ""
    for d in dataset_list:
        dataset = dataset + " " + d.split('/')[4]

    # with open('./leaderboard/experiments.csv', mode='a', newline='') as file:
    with open('./leaderboard/leaderboard_other.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([date, model_name, trial_number, dataset, optimizer, learning_rate, batch_size, scheduler, val_dice, val_loss])

def main_bs_lr():
    train_params = {
        'Date' : '07_23_24',
        'Trial Number' : 4,
        'Load Weights' : False,
        'Epochs' : 5,
        'Batch Size' : 2,
        'Optimizer' : 'SGD - momentum = 0.9',
        'Learning Rate' : 1e-5,
        'Augmentation Probability' : 0.5,
        'Loss Function' : 'composite_loss',
        'Monitor Param' : 'val_loss'
    }
    
    print("######################### TRAINING #########################")
    train(train_params)

    # for i in range(1, 7):
    #     print(f"######################### TRAINING {i} #########################")
    #     print(f"Batch Size: {train_params['Batch Size']}")
    #     train(train_params)
    #     train_params["Batch Size"] *= 2
    #     train_params["Trial Number"] += 1

def main_opt_lr():
    train_params = {
        'Date' : '07_23_24',
        'Trial Number' : 25,
        'Load Weights' : False,
        'Epochs' : 50,
        'Batch Size' : 2,
        'Optimizer' : 'default',
        'Learning Rate' : 1e-5,
        'Augmentation Probability' : 0.5,
        'Loss Function' : 'composite_loss',
        'Monitor Param' : 'val_loss'
    }

    opt = ['Adam', 'AdamW', 'SGD', 'SGD - 0.9']
    lr = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]

    for o in opt:
        for l in  lr:
            if((o == 'Adam' or o == 'AdamW') and (l == 0.1 or l == 0.01)):
                continue
            
            print(f"######################### TRAINING #{train_params['Trial Number']} #########################")
            print(f"Optimizer: {o}")
            print(f"Learning Rate: {l}")

            train_params['Optimizer'] = o
            train_params['Learning Rate'] = l
            train(train_params)

            train_params["Trial Number"] += 1

def main_tuple():
    train_params = {
        'Date' : '08_05_24',
        'Trial Number' : 1,
        'Load Weights' : False,
        'Epochs' : 50,
        'Batch Size' : 2,
        'Optimizer' : 'default',
        'Learning Rate' : 1e-5,
        'Scheduler' : 'none',
        'Augmentation Probability' : 0.5,
        'Loss Function' : 'composite_loss',
        'Monitor Param' : 'val_loss'
    }

    # train_combo = [
    #     ("Adam", 1e-05),
    #     ("Adam", 0.0001),
    #     ("Adam", 1e-06),
    #     ("AdamW", 0.0001),
    #     ("AdamW", 1e-05),
    #     ("AdamW", 1e-06),
    #     ("SGD", 0.01),
    #     ("SGD", 0.001),
    #     ("SGD", 0.0001),
    #     ("SGD - 0.9", 0.001),
    #     ("SGD - 0.9", 0.0001),
    #     ("SGD - 0.9", 0.01),
    #     ("SGD - 0.9", 1e-05)
    # ]

    # train_combo = [
        # ("SGD - 0.5", 0.01, "Linear")
        # ("AdamW", 1e-05, "ExponentialDecay"),
        # ("SGD - 0.5", 0.01, "ExponentialDecay"),
        # ("SGD - 0.75", 0.01, "ExponentialDecay")
    # ]

    # schedulers = ['Linear', 'ExponentialDecay', 'CosineDecay', 'CosineDecayRestarts']

    # batch_sizes = [1, 4, 8, 16, 32, 64]
    # batch_sizes = [2]


    '''for opt, lr in train_combo:
        for s in schedulers:
            print(f"######################### TRAINING #{train_params['Trial Number']} #########################")
            print(f"Optimizer: {opt}")
            print(f"Learning Rate: {lr}")
            print(f"Scheduler: {s}")

            if(train_params['Trial Number'] == 29):
                train_params["Trial Number"] += 1
                continue

            if(train_params['Trial Number'] == 49):
                train_params["Trial Number"] += 1
                continue

            if(train_params['Trial Number'] == 50):
                train_params["Trial Number"] += 1
                continue

            train_params['Optimizer'] = opt
            train_params['Learning Rate'] = lr
            train_params['Scheduler'] = s
            train(train_params)

            train_params["Trial Number"] += 1'''

    for opt, lr, s in train_combo:
        for b in batch_sizes:
            print(f"######################### TRAINING #{train_params['Trial Number']} #########################")
            print(f"Optimizer: {opt}")
            print(f"Learning Rate: {lr}")
            print(f"Scheduler: {s}")
            print(f"Batch Size: {b}")

            train_params['Optimizer'] = opt
            train_params['Learning Rate'] = lr
            train_params['Scheduler'] = s
            train_params['Batch Size'] = b
            train(train_params)

            train_params["Trial Number"] += 1

def main_fix_linear_sch_training():
    train_params = {
        'Date' : '08_03_24',
        'Trial Number' : 3,
        'Load Weights' : False,
        'Epochs' : 50,
        'Batch Size' : 2,
        'Optimizer' : 'default',
        'Learning Rate' : 1e-5,
        'Scheduler' : 'none',
        'Augmentation Probability' : 0.5,
        'Loss Function' : 'composite_loss',
        'Monitor Param' : 'val_loss'
    }

    # train_combo = [
    #     ("SGD - 0.5", 0.01, "Linear"),
    #     ("SGD - 0.75", 0.01, "Linear"),
    #     ("SGD - 0.9", 0.01, "Linear"),
    #     ("AdamW", 1e-05, "Linear"),
    #     ("Adam", 1e-05, "Linear"),
    #     ("SGD - 0.25", 0.01, "Linear"),
    #     ("SGD - 0.0", 0.01, "Linear"),
    #     ("SGD - 0.99", 0.0001, "Linear"),
    #     ("SGD - 0.95", 0.001, "Linear"),
    #     ("SGD - 0.9", 0.001, "Linear")
    # ]
    train_combo = [
        ("SGD - 0.5", 0.01, "Linear"),
        ("SGD - 0.75", 0.01, "Linear")
    ]

    # batch_sizes = [1, 4, 8, 16, 32, 64]
    batch_sizes = [1, 2, 4, 8, 16, 32, 64]

    for opt, lr, s in train_combo:
        for b in batch_sizes:
            print(f"######################### TRAINING #{train_params['Trial Number']} #########################")
            print(f"Optimizer: {opt}")
            print(f"Learning Rate: {lr}")
            print(f"Scheduler: {s}")
            print(f"Batch Size: {b}")

            if(train_params["Trial Number"] == 3):
                train_params["Trial Number"] += 1
                continue

            if(train_params["Trial Number"] == 4):
                train_params["Trial Number"] += 1
                continue
        
            train_params['Optimizer'] = opt
            train_params['Learning Rate'] = lr
            train_params['Scheduler'] = s
            train_params['Batch Size'] = b
            train(train_params)

            train_params["Trial Number"] += 1

# Most likely not needed b/c model.compile occurs twice
def main_fix_cosrest_sch_training():
    train_params = {
        'Date' : '08_06_24',
        'Trial Number' : 1,
        'Load Weights' : False,
        'Epochs' : 50,
        'Batch Size' : 2,
        'Optimizer' : 'default',
        'Learning Rate' : 1e-5,
        'Scheduler' : 'none',
        'Augmentation Probability' : 0.5,
        'Loss Function' : 'composite_loss',
        'Monitor Param' : 'val_loss'
    }

    # train_combo = [
    #     # use tupples in temp.txt
    #     ("SGD - 0.5", 0.01, "Linear"),
    # ]

    schedulers = ['CosineDecayRestarts']

    for opt, lr, s in train_combo:
        for s in schedulers:
            print(f"######################### TRAINING #{train_params['Trial Number']} #########################")
            # s = s + ' - ' + train_params['Epochs'] + ', ' + b
            print(f"Optimizer: {opt}")
            print(f"Learning Rate: {lr}")
            print(f"Scheduler: {s}")
            # print(f"Batch Size: {b}")
        
            train_params['Optimizer'] = opt
            train_params['Learning Rate'] = lr
            train_params['Scheduler'] = s
            train(train_params)

            train_params["Trial Number"] += 1

def main_250():
    # batch sizes: 1, 2, 4, 8, 16, 32, 64
    # optimizers: 'Adam', 'AdamW', 'SGD - '
    # learning rates: 0.1, 0.01, 0.001, 1e-4, 1e-5, 1e-6
    # schedulers: 'Linear', 'ExponentialDecay', 'CosineDecay', 'CosineDecayRestarts'

    train_params = {
        'Date' : '08_18_24',
        'Dataset' : '/data/satvik/data/tibia/train',
        'Trial Number' : 1,
        'Load Weights' : False,
        'Epochs' : 250,
        'Batch Size' : 2,
        'Optimizer' : 'AdamW',
        'Learning Rate' : 1e-05,
        'Scheduler' : 'ExponentialDecay',
        'Augmentation Probability' : 0.5,
        'Loss Function' : 'composite_loss',
        'Monitor Param' : 'val_loss'
    }

    # train_combo = [
        # ("SGD - 0.5", 0.01, "Linear")
        # ("AdamW", 1e-05, "ExponentialDecay"),
        # ("SGD - 0.5", 0.01, "ExponentialDecay"),
        # ("SGD - 0.75", 0.01, "ExponentialDecay")
    # ]

    print(f"######################### TRAINING #{train_params['Trial Number']} #########################")
    print(f"Optimizer: {train_params['Optimizer']}")
    print(f"Learning Rate: {train_params['Learning Rate']}")
    print(f"Scheduler: {train_params['Scheduler']}")
    print(f"Batch Size: {train_params['Batch Size']}")
    print(f"Dataset: {train_params['Batch Size']}")
    train(train_params)

def main_multiple_datasets():
    # batch sizes: 1, 2, 4, 8, 16, 32, 64
    # optimizers: 'Adam', 'AdamW', 'SGD - '
    # learning rates: 0.1, 0.01, 0.001, 1e-4, 1e-5, 1e-6
    # schedulers: 'Linear', 'ExponentialDecay', 'CosineDecay', 'CosineDecayRestarts'

    train_params = {
        'Date' : '08_18_24',
        'Dataset' : '/data/satvik/data/tibia/train',
        'Trial Number' : 1,
        'Load Weights' : False,
        'Epochs' : 250,
        'Batch Size' : 2,
        'Optimizer' : 'AdamW', #'AdamW',
        'Learning Rate' : 1e-5 , # 0.01
        'Scheduler' : 'Linear',
        'Augmentation Probability' : 0.5,
        'Loss Function' : 'composite_loss',
        'Monitor Param' : 'val_loss'
    }

    datasets = [
        '/data/satvik/data/tibia/train',
        # '/data/satvik/data/tibia/test'
    ]

    for d in datasets:
        print(f"######################### TRAINING #{train_params['Trial Number']} #########################")
        print(f"Optimizer: {train_params['Optimizer']}")
        print(f"Learning Rate: {train_params['Learning Rate']}")
        print(f"Scheduler: {train_params['Scheduler']}")
        print(f"Batch Size: {train_params['Batch Size']}")
        print(f"Dataset: {d}")

        train_params['Dataset'] = d

        train(train_params)
        train_params["Trial Number"] += 1

def main_opt():
    train_params = {
        'Date' : '08_24_24',
        'Dataset' : ['/data/satvik/data_2023/new_data_110623/train',
                     '/data/satvik/data/liver/train',
                     '/data/satvik/data/tibia/train'],
        'Trial Number' : 1,
        'Load Weights' : False,
        'Epochs' : 250,
        'Batch Size' : 2,
        'Optimizer' : 'AdamW', #'AdamW',
        'Learning Rate' : 0.01 , # 1e-5
        'Scheduler' : 'Linear',
        'Augmentation Probability' : 0.5,
        'Loss Function' : 'composite_loss',
        'Monitor Param' : 'val_loss'
    }

    # schedulers = ['Linear', 'ExponentialDecay']
    optimizers = ['SGD - 0.5', 'SGD - 0.75', 'SGD - 0.9']
    # learning_rates = []
    datasets = [
                ['/data/satvik/data/liver/train'],
                ['/data/satvik/data/tibia/train']
                ]

    # for s in schedulers:
    #     print(f"######################### TRAINING #{train_params['Trial Number']} #########################")
    #     train_params['Scheduler'] = s

    #     print(f"Optimizer: {train_params['Optimizer']}")
    #     print(f"Learning Rate: {train_params['Learning Rate']}")
    #     print(f"Scheduler: {train_params['Scheduler']}")
    #     print(f"Batch Size: {train_params['Batch Size']}")

    #     train(train_params)
    #     train_params["Trial Number"] += 1

    for d in datasets:
        for o in optimizers:
            print(f"######################### TRAINING #{train_params['Trial Number']} #########################")
            train_params['Dataset'] = d
            train_params['Optimizer'] = o

            print(f"Dataset: {train_params['Dataset']}")
            print(f"Optimizer: {train_params['Optimizer']}")
            print(f"Learning Rate: {train_params['Learning Rate']}")
            print(f"Scheduler: {train_params['Scheduler']}")
            print(f"Batch Size: {train_params['Batch Size']}")

            train(train_params)
            train_params["Trial Number"] += 1

if __name__ == '__main__':
    # main_bs_lr()
    # main_opt_lr()
    # main_tuple()
    # main_fix_linear_sch_training()
    # main_250()
    # main_multiple_datasets()
    main_opt()