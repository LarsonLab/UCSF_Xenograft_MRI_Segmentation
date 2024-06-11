#HOW TO RUN:python main.py
#use environment: /home/erdiaz/miniconda3/envs/tf


import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from UCSF_Prostate_Segmentation.old.metrics import dice_coef
import tensorflow as tf
from UCSF_Prostate_Segmentation.old.plot import save_plots
from UCSF_Prostate_Segmentation.old.r2udensenet_1d import r2udensenet
'''from dense_unet import denseunet
from unet import unet_model
from res_unet import resunet
from new_r2unet import r2unet'''
from scipy import io as sio
from UCSF_Prostate_Segmentation.old.data2D_ucsf_1d import load_train_data, load_test_data
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
from sklearn.preprocessing import binarize
from sklearn.utils import class_weight
import pdb
from sklearn.model_selection import KFold
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from time import time
import warnings
warnings.filterwarnings("ignore")




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

#################### Training of the Network by Five Fold Cross Validation #################################
# def return_directories(date, trial_num):
#         weight_save_directory = './weights_2023/weights_' + date + '/trial_' + str(trial_num)
#         print(f'weight_save_directory: {weight_save_directory}')
       
#         log_directory = './logs_2023/Training_' + date + '/trial_' + str(trial_num)
#         print(f'log_directory: {log_directory}')
#         print('')

#         if not os.path.exists(weight_save_directory):
#             print("DOESN'T EXIST")
#             os.makedirs(weight_save_directory)
#             print(f'Directory created: {weight_save_directory}')
#         if not os.path.exists(log_directory):
#             print("DOESN'T EXIST")
#             os.makedirs(log_directory)
#             print(f'Directory created: {log_directory}')
        
#         return weight_save_directory, log_directory



def get_model_name(k,e):
        return 'model_r2udensenet'+str(k)+'_'+str(e)+'.hdf5'
def get_log_name(k):
        return 'log_r2udensenet'+str(k)+'.csv'

def generate_generator(generator, images_train, mask_train):
        train_gen = generator.flow(images_train, y=None, batch_size = 2, shuffle = True, seed = 100, 
                                              sample_weight = None)
    
        label_gen = generator.flow(mask_train, y=None, batch_size = 2, shuffle = True, seed = 100, 
                                              sample_weight = None)
        
        while True:
            reg_image = train_gen.next()
            mask = label_gen.next()
            # image_final = np.append(None,reg_image,axis=3)
            # Added
            if np.unique(mask).shape[0] > 2:
                mask[mask >= 0.5] = 1.
                mask[mask < 0.5] = 0.
            #
            yield reg_image,mask        

def train(bool_load_weights = False):
    
    def return_directories(date):
        #        out_dir = log_directory + "/plots"+str(fold_num)
        # if not os.path.exists(out_dir):
        #     os.makedirs(out_dir)     

        # weight_save_directory = './weights' + date + '/trial'
        # print(f'weight_save_directory: {weight_save_directory}')
        log_directory_main = './logs'
        log_directory_test = log_directory_main + '/Test2'
        # weight_save_directory = log_directory_test+ '/weights'
        # log_directory = './logs_2023/Training_' + date + '/trial_' + str(trial_num)
        # os.path.join(log_directory,weight_save_directory)
        # print(f'log_directory: {log_directory}')
        # print('')

        if not os.path.exists(log_directory_main):
            print("DOESN'T EXIST")
            os.makedirs(log_directory_main)
            print(f'Directory created: {log_directory_main}')    
        if not os.path.exists(log_directory_test):
            print("DOESN'T EXIST")
            os.makedirs(log_directory_test)
            print(f'Directory created: {log_directory_test}')
        # if not os.path.exists(weight_save_directory):
        #     print("DOESN'T EXIST")
        #     os.makedirs(weight_save_directory)
        #     print(f'Directory created: {weight_save_directory}')
        
        return log_directory_main,log_directory_test
    
    date = None
    log_directory_main,log_directory_test= return_directories(date)
    kfold = KFold(n_splits = 5, shuffle = True, random_state = 1)
    # batch size
    # batch_s = 1
    # pdb.set_trace()
    # print(f'kfold: {kfold}')

    images_train, mask_train = load_train_data()
    images_train = images_train.astype('float64')
    mask_train = mask_train.astype('float64')

    # Normalization
    images_train_mean = np.mean(images_train)
    images_train_std = np.std(images_train)
    images_train = (images_train - images_train_mean)/images_train_std
    
    # Converts masks from 0/255 to 0/1
    mask_train /= 255.

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
    augmentation_datagen_train = ImageDataGenerator(
                    rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    vertical_flip=True,
                    fill_mode='nearest'
                    )
   
    augmentation_probability = 0.2 
    def custom_generator(generator, augmentation_probability):
        for x_batch, y_batch in generator:
            for i in range(len(x_batch)):
                if np.random.random() < augmentation_probability:
                    x_batch[i] = augmentation_datagen_train.random_transform(x_batch[i])
                    
            yield x_batch, y_batch



    #
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
    
    # datagen_val = ImageDataGenerator(
    #                 # rotation_range=0.2,
    #                 # width_shift_range=0.05,
    #                 # height_shift_range=0.05,
    #                 # shear_range=0.05,
    #                 # zoom_range=0.05,
    #                 # horizontal_flip=True,
    #                 # vertical_flip=True,
    #                 # fill_mode='nearest'
    #                 )

    # def generate_generator(generator, images_train, mask_train):
    #     train_gen = generator.flow(images_train, y=None, batch_size = 2, shuffle = True, seed = 100, 
    #                                           sample_weight = None)
    
    #     label_gen = generator.flow(mask_train, y=None, batch_size = 2, shuffle = True, seed = 100, 
    #                                           sample_weight = None)
        
    #     while True:
    #         image_T2 = train_gen.next()
    #         mask = label_gen.next()
    #         # Added
    #         if np.unique(mask).shape[0] > 2:
    #             mask[mask >= 0.5] = 1.
    #             mask[mask < 0.5] = 0.
    #         #
    #         yield image_T2,mask

    # train_generator = generate_generator(image_datagen, mask_datagen, images_train, mask_train)
    # def get_model_name(k):
    #     return 'model_r2udensenet'+str(k)+'.hdf5'
    
    # def get_log_name(k):
    #     return 'log_r2udensenet'+str(k)+'.csv'

    # pdb.set_trace()
    print(f'kfold.split(images_train): {kfold.split(images_train)}')

    Images = kfold.split(images_train)
    fold_num = 1
    for train_index, validation_index in Images:

        # pdb.set_trace()
        # print(f'train_index: {train_index}')
        # # print(f'train_index: {train_index[0]} - {train_index[-1]}')
        # print(f'train_index.shape: {train_index.shape}')
        # print(f'train_index.dtype: {train_index.dtype}')
        # print(f'validation_index: {validation_index}')
        # # print(f'validation_index: {validation_index[0]} - {validation_index[-1]}')
        # print(f'validation_index.shape: {validation_index.shape}')
        # print(f'validation_index.dtype: {validation_index.dtype}')

        trainData = images_train[train_index] #images
        trainMask = mask_train[train_index] #groundtruth

    
        validationData = images_train[validation_index]#val images
        validationMask = mask_train[validation_index]#ground truth 

        # image_data = validationData
        # ground_truth_data = validationMask

        train_generator_temp = generate_generator(datagen_no_augmentation, trainData, trainMask)
        train_generator = custom_generator(train_generator_temp, augmentation_probability)
        validation_generator = generate_generator(datagen_no_augmentation, validationData, validationMask)
      
        # train_generator = generate_generator(datagen_train, trainData, trainMask)
        # validation_generator = generate_generator(datagen_val, validationData, validationMask)

        # plt.title("test")
        # image = validationData[0]
        # image2 = validationMask[0]
        # plt.imshow(image[:,:,0], cmap='gray')
        # # plt.imshow(image2[:,:,0], alpha=0.4)  
        # plt.savefig(f'{log_directory_main}/test3.png')
        # plt.savefig(f)
        model = r2udensenet()
        # load weights
        # unlock model
        # ImageSave(model,validationMask,log_directory,None)
        
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

        # if bool_load_weights:
        #   load_date = '8_11_23'
        #   load_trial_number = '3'
        #   model_number = '2'
        #   weight_load_directory = './weights_2023/weights_' + load_date + '/trial_' + str(load_trial_number)
        # model_num = 1
        # model_name = '/model/'+ 'model_r2udensenet' + str(model_num) + '.hdf5'
        # if not os.path.exists(model_name):
        #     print("DOESN'T EXIST")
        #     os.makedirs(model_name)
        #     print(f'Directory created:{model_name}')
        # model_num = model_num + 1
        # model.load_weights(weight_save_directory + model_name)
        # model.trainable = True
        #   print(f'########## Weights Loaded from {model_name} ##########')
        #   print("weights:", len(model.weights))
        #   print("trainable_weights:", len(model.trainable_weights))
        #   print("non_trainable_weights:", len(model.non_trainable_weights))
        # else:
        #   print('########## NO Weights Loaded ##########')

        # print('########## Summary ##########')
        # model.summary()
        # for debugging
        # pdb.set_trace()

        plot_dir = log_directory_test + "/plots"+str(fold_num)
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        weight_dir = plot_dir+ "/weights"
        if not os.path.exists(weight_dir):
            os.makedirs(weight_dir)

        # model_checkpoint2 = ModelCheckpoint(os.path.join(weight_dir,"saved-model-{epoch:02d}.hdf5"),
        #  monitor = 'val_loss',
        #  save_best_only = False,
        #  mode = max,
         
        # )
        model_checkpoint = ModelCheckpoint(os.path.join( weight_dir,"saved-model-{epoch:02d}-{val_loss:.2f}.hdf5"),
            # monitor = 'loss',
            monitor = 'val_loss',
            save_freq = 'epoch',
            verbose = 1,
            mode = 'min', # min because monitoring loss
            save_best_only = False)
    
        logger = CSVLogger(os.path.join(log_directory_test, get_log_name(fold_num)),
            separator = ',',
            append = False)
    
        # pdb.set_trace()
        # print('##### Before model training #####')

        start = time()
        history = model.fit_generator(
            train_generator,
            steps_per_epoch = len(trainData)/2, # '/2' = bactch size
            epochs = 2, #25
            verbose = 1,
            validation_data = validation_generator,
            validation_steps = len(validationData)/2 , # 26 should be num validation data / batchsize
            # class_weight = class_weights_dict, #class_weights_dict, {0.0: 0.5, 1.0: 10}
            # sample_weight = class_weights_dict,
            callbacks = [model_checkpoint,logger])
        print("===== K Fold Validation Step ======", fold_num)
        
          # image = validationData[0]
        # image2 = validationMask[0]
        # plt.imshow(image[:,:,0], cmap='gray')
        # # plt.imshow(image2[:,:,0], alpha=0.4)
  
             
        # out_dir = log_directory_test + "/plots"+str(fold_num)
        # if not os.path.exists(out_dir):
        #     os.makedirs(out_dir)
        
 
        train_comp_loss = history.history['loss']
        val_comp_loss = history.history['val_loss']
        train_dice_coef = history.history['dice_coef']
        val_dice_coef = history.history['val_dice_coef']
        train_dice_loss = history.history['dice_loss']
        val_dice_loss = history.history['val_dice_loss']
        train_iou = history.history['IoU']
        val_iou = history.history['val_IoU']
        # num_epochs = history.history['epochs']
        save_plots(
        fold_num,
        train_comp_loss,
        val_comp_loss, 
        train_dice_coef,
        val_dice_coef,   
        train_dice_loss,
        val_dice_loss,
        train_iou,
        val_iou, 
        plot_dir
         )
        fold_num = fold_num + 1

        out_epochs = plot_dir + "/epochs"
        if not os.path.exists(out_epochs):
          os.makedirs(out_epochs)

        epoch = []    
        filepath = os.listdir(weight_dir)
        for file in filepath:
            if file.endswith('hdf5'):
                epoch.append(file)
        print(epoch[0])
        #find average rates
        for i in history.epoch:
         
         for j in range(3):
          image_data = validationData[0]
          ground_truth_data = validationMask[0]

          model.load_weights(os.path.join(weight_dir,epoch[i]))
          predict = model.predict(np.expand_dims(image_data,axis=0))[0]
  
          plt.figure(figsize=(12,6))   

          plt.subplot(3,2,(j*2)+1)
          plt.imshow(image_data,cmap = 'gray')
          plt.imshow(ground_truth_data,alpha=0.4)
          plt.title('Ground Truth Mask Overlay')
            
          plt.subplot(3,2,(j+1)*2)
          plt.imshow(image_data,cmap = 'gray')
          plt.imshow(predict,alpha=0.4)
          plt.title("Predicted Mask Overlay")
        plt.savefig(f'{out_epochs}/e_'+str(i+1)+'.png')

 
        
        # plt.title(f'Model #{fold_no}')
        # # plt.ylim(0, 1)
        # plt.plot(history.history['loss'])  
        # plt.plot(history.history['val_loss'])  
        # plt.ylabel('Composite Loss(75/25)',fontweight='bold')  
        # plt.xlabel('Epochs',fontweight='bold')  
        # plt.legend(['Train', 'Validation'], loc='upper left')
        # plt.savefig(f'{out_dir}/Composite_loss_{fold_no}.png')
        # plt.close()
        
        # plt.title(f'Model #{fold_no}')
        # # plt.ylim(0, 1)
        # plt.plot(history.history['dice_coef'])  
        # plt.plot(history.history['val_dice_coef'])  
        # plt.ylabel('Dice Coefficient',fontweight='bold')  
        # plt.xlabel('Epochs',fontweight='bold')  
        # plt.legend(['Train', 'Validation'], loc='upper left')
        # plt.savefig(f'{out_dir}/Dice_coef_{fold_no}.png')
        # plt.close()

        # ## FIX
        # plt.title(f'Model #{fold_no}')
        # # plt.ylim(0, 1)
        # plt.plot(history.history['dice_loss'])  
        # plt.plot(history.history['val_dice_loss'])  
        # plt.ylabel('Dice Loss',fontweight='bold')  
        # plt.xlabel('Epochs',fontweight='bold')  
        # plt.legend(['Train', 'Validation'], loc='upper left')
        # plt.savefig(f'{out_dir}/Dice_loss_{fold_no}.png')
        # plt.close()

        # # plt.title(f'Model #{fold_no}')
        # # # plt.ylim(0, 1)
        # # plt.plot(history.history['weighted_bce'])  
        # # plt.plot(history.history['val_weighted_bce'])  
        # # plt.ylabel('Weighted Binary Cross-Entropy Loss',fontweight='bold')  
        # # plt.xlabel('Epochs',fontweight='bold')  
        # # plt.legend(['Train', 'Validation'], loc='upper left')
        # # plt.savefig(f'{plot_directory}/BCE_loss_{fold_no}.png')
        # # plt.close()

        # plt.title(f'Model #{fold_no}')
        # # plt.ylim(0, 1)
        # plt.plot(history.history['IoU'])  
        # plt.plot(history.history['val_IoU'])  
        # plt.ylabel('IoU',fontweight='bold')
        # plt.xlabel('Epochs',fontweight='bold')  
        # plt.legend(['Train', 'Validation'], loc='upper left')
        # plt.savefig(f'{out_dir}/IoU_{fold_no}.png')
        # plt.close()

        # print(f'Plots saved as in {plot_directory}')
        


        
        
        # val_punch = np.append(validationData_T2,validationData_T1, axis = 3)
        val_punch = validationData
        scores = model.evaluate(val_punch,validationMask, verbose=0)
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
        
     

        

    #     #####
    #     print(f'val_punch.dtype: {val_punch.dtype}')
    #     print(f'val_punch.shape: {val_punch.shape}')
    #     print(f'length of scores: {len(scores)}')
    #     print(f'composite loss: {composite_loss}')
    #     print(f'dice_loss: {dice_loss}')
    #     print(f'dice: {dice}')
    #     print(f'precision: {precision}')
    #     print(f'recall: {recall}')
    #     print(f'auc: {auc}')
    #     print(f'accuracy: {accuracy}')
    #     print(f'IoU: {IoU_test}')
    #     print(f'binaryCE_original: {binaryCE_original}')
    #     print(f'bce: {bce}')
    #     #####

    # print('Composite Loss == ', composite_loss , 'Mean Composite Loss == ', np.mean(composite_loss))
    # print('Dice Loss == ', dice_loss , 'Mean Dice Loss == ', np.mean(dice_loss))
    # print('DICE Coefficient == ', dice , 'Mean Dice == ', np.mean(dice))
    # print('Precision ==', precision, 'Mean Precision ==', np.mean(precision))
    # print('Recall ==', recall, 'Mean Recall ==', np.mean(recall))
    # print('AUC ==', auc, 'Mean AUC ==', np.mean(auc))
    # print('Accuracy ==', accuracy, 'Mean Accuracy ==',np.mean(accuracy))
    # print('IoU ==', IoU_test, 'Mean IoU ==', np.mean(IoU_test))
    # print('binaryCE_original ==', binaryCE_original, 'Mean binaryCE_original ==',np.mean(binaryCE_original))
    # print('bce ==', bce, 'Mean bce ==',np.mean(bce))
           
if __name__ == '__main__':
    train(bool_load_weights = False)
#     date = '8_25_23'
#     trial_number = 2
    bool_load_weights = False

#     print('######################### TRAINING #########################')
#     train(date, trial_number, bool_load_weights)

#add dice coef
#