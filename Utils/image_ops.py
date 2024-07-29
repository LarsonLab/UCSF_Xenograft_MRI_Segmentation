from tqdm import tqdm
import numpy as np 
import scipy.ndimage as ndi 
import torch 
from torch import Tensor 
from torchvision import transforms 
from typing import Callable, BinaryIO, Match,Pattern,Tuple,Union, Optional
from Metrics.utils import one_hot2dist, class2one_hot
from functools import partial, reduce 
from operator import itemgetter, mul 
from PIL import Image, ImageOps
from torchvision.transforms.v2 import functional as F 
from torchvision import transforms 


D = Union[Image.Image,np.ndarray,Tensor]



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

def dist_map_transform(resolution: Tuple[float, ...], K: int) -> Callable[[D], Tensor]:
        return transforms.Compose([
                gt_transform(resolution, K),
                lambda t: t.cpu().numpy(),
                partial(one_hot2dist, resolution=resolution),
                lambda nd: torch.tensor(nd, dtype=torch.float32)
        ]) 

def gt_transform(resolution: Tuple[float, ...], K: int) -> Callable[[D], Tensor]:
        return transforms.Compose([
                lambda img: np.array(img)[...],
                lambda nd: torch.tensor(nd, dtype=torch.int64)[None, ...],  # Add one dimension to simulate batch
                partial(class2one_hot, K=K),
                itemgetter(0)  # Then pop the element to go back to img shape
        ])

class RandomRotationWithPadding:
     
    def __init__(self,degrees):
        self.degrees = degrees 

    def __call__(self,img):
        angle = torch.empty(1).uniform_(-self.degrees,self.degrees).item()
        return self.rotate_with_padding(img,angle)
    
    def rotate_with_padding(img,angle):
        _,h,w = img.shape
        diagonal = int((w**2 + h**2)**0.5)
        padding = (diagonal - w) // 2
        img_padded = F.pad(img,padding,fill=0)
        img_rotated = F.rotate(img_padded,angle)
        #center crop the image back to the original size 
        img_cropped = F.center_crop(img_rotated,[h,w])


def get_binary_labels(masks):

    binary_labels = []
    for i, mask in enumerate(masks):
        j = np.max(masks[i])
        if j > 0:
            binary_labels.append(1)
        else: 
            binary_labels.append(0)
    return binary_labels
     

    



