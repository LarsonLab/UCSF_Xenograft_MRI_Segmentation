"""
Data loader for loading data from muptile datasets
"""

import os 
import numpy as np
import h5py
import pdb

def load_train_data(paths):
    # Pass in paths when calling data loader
    # paths = [
    #     '/data/satvik/data_2023/new_data_110623/train',
    #     '/data/satvik/data/liver/train',
    #     '/data/satvik/data/tibia/train'
    # ]

    first_bool = True
    for path in paths:
        print(f"Loading data from: {path}")
        for file in os.listdir(path):
            if(file.endswith('.h5')):
                temp = h5py.File(os.path.join(path, file), 'r')
                img = np.array(temp['img'])
                mask = np.array(temp['mask'])
                # print(f'img size: {img.shape}')
                # print(f'mask size: {mask.shape}')

                if(img.shape[0] != 192 or mask.shape[0] != 192):
                    print(f"file: {file}")
                    print(f'img size: {img.shape}')
                    print(f'mask size: {mask.shape}')
                    continue

                if(first_bool):
                    images_train = img
                    mask_train = mask
                    first_bool = False
                else:
                    images_train = np.append(images_train, img, axis=-1)
                    mask_train = np.append(mask_train, mask, axis=-1)

    images_train = np.moveaxis(images_train, [-1], [0])
    images_train = np.expand_dims(images_train, axis=-1)
    mask_train = np.moveaxis(mask_train, [-1], [0])
    mask_train = np.expand_dims(mask_train, axis=-1)

    print('=============== Loading of UCSF Training Images and Masks ===================')
    return images_train,mask_train

def load_test_data(paths):
    # Pass in paths when calling data loader
    # paths = [
    #     '/data/satvik/data_2023/new_data_110623/test',
    #     '/data/satvik/data/liver/test',
    #     '/data/satvik/data/tibia/test',
    # ]
    
    first_bool = True
    for path in paths:
        print(f"Loading data from: {path}")
        for file in os.listdir(path):
            if(file.endswith('.h5')):
                temp = h5py.File(os.path.join(path, file), 'r')
                img = np.array(temp['img'])
                mask = np.array(temp['mask'])
                # print(f'img size: {img.shape}')
                # print(f'mask size: {mask.shape}')

                # If some data is the wrong size (temporary fix)
                # if(img.shape[0] != 192 or mask.shape[0] != 192):
                #     print(f"file: {file}")
                #     continue

                if(first_bool):
                    images_test = img
                    mask_test = mask
                    first_bool = False
                else:
                    images_test = np.append(images_test, img, axis=-1)
                    mask_test = np.append(mask_test, mask, axis=-1)
    
    images_test = np.moveaxis(images_test, [-1], [0])
    mask_test = np.moveaxis(mask_test, [-1], [0])
    
    mask_test = mask_test.astype(dtype=bool)

    images_test = np.expand_dims(images_test, axis=-1)

    print('======Loading of UCSF Test Data=======')
    return images_test, mask_test

# For debugging
# images_train, mask_train = load_train_data()
# print(f'mask_train: {mask_train.shape}')
# print(f'images_train: {images_train.shape}')
# print(f'mask_train.dtype: {mask_train.dtype}')
# print(f'images_train.dtype: {images_train.dtype}')
# print(f'np.unique(mask_train, return_counts=True): {np.unique(mask_train, return_counts=True)}')
# print(f'np.unique(images_train, return_counts=True): {np.unique(images_train, return_counts=True)}')

# stackedArr, mask_test = load_test_data()
# print(f'stackedArr: {stackedArr.shape}')
# print(f'mask_test: {mask_test.shape}')
# print(f'stackedArr.dtype: {stackedArr.dtype}')
# print(f'mask_test.dtype: {mask_test.dtype}')
# print(f'np.unique(mask_test, return_counts=True): {np.unique(mask_test, return_counts=True)}')
