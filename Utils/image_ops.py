from tqdm import tqdm
import numpy as np 
import scipy.ndimage as ndi 


def resize_images(data,new_dim,interpolation_order):
    if data.shape[1] < 4:
        x_shape = data.shape[2]
        y_shape = data.shape[3]
        zoomed_data = np.zeros(data.shape[0],data.shape[1],new_dim,new_dim)
        zoom_factor = (1,(new_dim / x_shape),(new_dim / y_shape))
    else: 
        x_shape = data.shape[1]
        y_shape = data.shape[2]
        zoomed_data = np.zeros((data.shape[0],new_dim,new_dim,data.shape[3]))
        zoom_factor = ((new_dim / x_shape),(new_dim / y_shape),1)
    counter = 0 
    for image in tqdm(data,desc='Resizing Data',unit='images'): 
        new_image = ndi.zoom(image,zoom_factor,order=interpolation_order)
        zoomed_data[counter] = new_image
        counter += 1
    return zoomed_data


def threshold_image(image,threshold): 

    if not isinstance(image,np.ndarray):
        image = image.numpy()
    condition = image > threshold
    thresholded_image = np.where(condition,1,0).astype(np.uint8)
    return thresholded_image

