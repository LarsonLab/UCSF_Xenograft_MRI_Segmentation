import os 
import numpy as np
import h5py
import pdb

def load_train_data():
    path = '/data/ernesto/data_2023/'
    file = 'ucsf_data_train_all.hdf5'

    data = h5py.File(os.path.join(path, file), 'r')
    images_train = np.array(data['img'])
    mask_train = np.array(data['mask'])
    
    print(images_train.shape)
    print(mask_train.shape)  
    print('========== Loading of ALL UCSF Training Data ==============')
    return images_train,mask_train

def load_test_data():
    path = '/data/ernesto/data_2023'
    file = 'ucsf_data_test_all.hdf5'
    
    data = h5py.File(os.path.join(path, file), 'r')
    images_test = np.array(data['img'])

    mask_test = np.array(data['mask'])
    mask_test = mask_test.astype(dtype=bool)

  
    print(images_test.shape)
    print(mask_test.shape)
    print('====== Loading of ALL UCSF Test Data with Tumors =======')
    return images_test,mask_test

# images_train, mask_train = load_train_data()
# print(f'images_train: {images_train.shape}')
# print(f'mask_train: {mask_train.shape}')
# print(f'images_train.dtype: {images_train.dtype}')
# print(f'mask_train.dtype: {mask_train.dtype}')
# print(f'np.unique: {np.unique(mask_train, return_counts=True)}')

# images_test, mask_test = load_test_data()
# print(f'images_test: {images_test.shape}')
# print(f'mask_test: {mask_test.shape}')
# print(f'images_test.dtype: {images_test.dtype}')
# print(f'mask_test.dtype: {mask_test.dtype}')
# print(f'np.unique: {np.unique(mask_test, return_counts=True)}')