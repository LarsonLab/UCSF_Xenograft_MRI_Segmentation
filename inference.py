import numpy as np 
import pandas as pd 
import torch 
import torchvision
from torchvision import utils as vutils 
from torchvision import transforms 
from torchvision.transforms import v2
from torch import nn 
from torch.nn import functional as F 
from torch.utils import data 
from torch.optim import SGD, Adam
from torch.optim import lr_scheduler as lr_scheduler 
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn import PoissonNLLLoss
from PIL import Image 
import matplotlib.pyplot as plt 
import os 
import csv 
from statistics import mean 
from architectures.torch_unet import UNet
from architectures.Attention_UNet import Attention_UNet, R2Attention_Unet
from architectures.Mamba_Unet import LightMUNet
from Utils.data2D_ucsf_1d import load_train_data, load_test_data
from Utils.image_ops import threshold_image, dist_map_transform,RandomRotationWithPadding
from Metrics.plot import save_plots2, save_plots3
import sklearn 
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import datetime
import random 
from tqdm import tqdm
import datetime 
from Metrics.losses import DiceLoss, DiceBCELoss, IoULoss, TverskyIoULoss, BoundaryIoULoss, CompositeBoundaryLoss, TverskyBoundaryLoss, CompositeTversky,BoundaryTversky,BCE_BoundaryLoss, BoundaryLoss
from Metrics.losses import TverskyLoss
from testing import run_testing, MCDropout_Testing
import schedulefree
from Utils.image_ops import resize_images
from architectures.UNetR import create_UNetR
from Utils import h5_parsing
import h5py
from torchvision.models import resnet18

n_files = 0 
current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
detection_weights = '/home/henry/UCSF_Prostate_Segmentation/Weights/Resnet18_weights/2024-07-29_00-10-40.pth'
# segmentation_weights = '/home/henry/UCSF_Prostate_Segmentation/Weights/Attention_weights/2024-07-28_06-19-33_#_200cbound.pth'
segmentation_weights = '/home/henry/UCSF_Prostate_Segmentation/Weights/UNet_weights/2024-07-26_20-37-33_#_200.pth'

IMAGES_PATH = input('\n###NOTE###\nH5 FILES SHOULD BE COMBINED INTO AN HDF5 FILE\nIMAGES SHOULD BE PLACED IN img SUBDIRECTORY\n\nPLEASE ENTER IMAGES PATH: ')
SAVE_PATH = input('PLEASE ENTER SAVE PATH: ')
print()

def load_train_data(train_data_path):

        data = h5py.File(train_data_path, 'r')
        images_train = np.array(data['img'])
        
        return images_train 

def save_model_weights_path (path): 
    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_path = os.path.join(path,f'{current_time}_#_.pth')
    return save_path 

def check_if_trailing_slash(path): 
    if path[-1] == '/':
        return path[:-1] 
    else: 
        return path 
    
def normalization(data):
    data_min = np.min(data)
    data_max = np.max(data)
    data_normalized = ((data - data_min)/(data_max - data_min)) * 255
    return data_normalized 


class PredictionLoader(data.Dataset):

    def __init__(self,inputs,transform=None):
        self.inputs = inputs 
        self.transform = transform 
        self.input_dtype = torch.float32

    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self,index):
        image = self.inputs[index]
        x = torch.from_numpy(np.transpose((np.array(image)),(2,0,1))).type(self.input_dtype)

        return x  
    
def generate_prediction_dataset():

    images = load_train_data(IMAGES_PATH)
    print(f'\nLoading images from {IMAGES_PATH}\n')
    print(f'Image array shape: {images.shape}')
    print(f'train images min/max: {np.min(images)}/{np.max(images)}')

    image_shape = images.shape[1]
    for i in range(images.shape[0]):
        norm_img = normalization(images[i])
        images[i] = norm_img

    sanity_check(images)
    image_data_loader = [(images[i]) for i in range(len(images)-1)]
    image_data = PredictionLoader(image_data_loader)

    return image_data

def prediction_loaders(image_dataset,batch_size=1):
    prediction_loader = data.DataLoader(dataset=image_dataset,batch_size=batch_size,shuffle=False)
    return prediction_loader 

def loss_computations(image,mask): 

    loss_dice = DiceLoss()(image,mask)
    loss_composite = DiceBCELoss()(image,mask)
    loss_iou = IoULoss()(image,mask)

    dice_item = loss_dice.item()
    composite_item = loss_composite.item()
    iou_item = loss_iou.item()

    return dice_item,composite_item,iou_item 

def create_resnet18():
    net = resnet18()
    net.conv1 = nn.Conv2d(1,64,kernel_size=(7,7),stride=(2,2),padding=(3,3),bias=False)
    net.fc = nn.Linear(net.fc.in_features,2)
    return net 

def sanity_check(images): 

    fig, axes = plt.subplots(nrows=4,ncols=1,figsize=(15,15))
    fig.suptitle(f'{current_time}')
    for i in range(0,4): 
        rand = np.random.randint(0,len(images)-1)
        axes[i].imshow(images[rand],cmap='gray')
    plt.savefig("/home/henry/UCSF_Prostate_Segmentation/Data_plots/Sanity_checks/check2")
def inference(segmentation_model,segmentation_weights,detection_model,
              detection_weights,image_loader,device,clear_mem=False):
    num_positive_samples = 0 
    num_negative_samples = 0 
    unsorted_images = []
    unsorted_masks = []
    confident_images = []
    confident_masks = []
    unconfident_images = []
    unconfident_masks = []
    avg_std_dev = []
    detection_model = detection_model 
    segmentation_model = segmentation_model 
    detection_model.load_state_dict(torch.load(detection_weights))
    segmentation_model.load_state_dict(torch.load(segmentation_weights))
    detection_model = detection_model.to(device)
    detection_model.eval()
    segmentation_model = segmentation_model.to(device)
    segmentation_model.eval()
    positive_state = True
    with torch.no_grad():
        for img in tqdm(image_loader,total=len(image_loader),desc='Segmenting Dataset'):
            image = img.float().to(device)
            image_size = image.shape[2]
            classification1 = torch.argmax(detection_model(image))
            classification2 = torch.argmax(detection_model(image))
            classification3 = torch.argmax(detection_model(image))
            positive_state = 1 if ((classification1 + classification2 + classification3)/3) > 0.4 else 0 
            if positive_state == 1:  
                mcd_counter = 0 
                image_blank = []
                while mcd_counter < 51:
                    print(f'image input size: {image.shape}')
                    print(f'image input range: {torch.min(image)}/{torch.max(image)}')
                    output = segmentation_model(image) #(1,1,192,192)
                    print(f'output size: {output.shape}')
                    print(f'raw output range: {torch.min(output)}/{torch.max(output)}')
                    image_blank.append(output) #(1,192,192)
                    mcd_counter +=1 

                image_tensor = torch.cat(image_blank,dim=0) #(50,1,192,192)
                print(f'image tensor shape: {image_tensor.shape}')
                final_output = torch.mean(image_tensor,dim=0,keepdim=True) #(1,192,192)
                print(f'final output shape: {final_output.shape}')
                print(f'final output range: {torch.min(final_output)}/{torch.max(final_output)}')
                mean_image = torch.clamp(final_output,0.0,1.0)  #(1,1,192,192)
                unsorted_images.append(image)
                unsorted_masks.append(mean_image)
                mean_tensor = mean_image.expand_as(image_tensor)
                print(f'mean expanded tensor shape: {mean_tensor.shape}')
                squared_diff = (image_tensor - mean_tensor) ** 2
                variance = torch.mean(squared_diff,dim=0)
                standard_deviation = torch.sqrt(variance)
                print(f'standard deviation shape: {standard_deviation.shape}')
                avg_std_deviation = torch.mean(standard_deviation)
                avg_std_dev.append(avg_std_deviation.item())
                print(f'final std estimate: {avg_std_deviation.item()}')

            else: 
                avg_std_dev.append(0)
                blank_output = torch.zeros(1,1,image_size,image_size).to(device)
                print(f'image shape: {image.shape}')
                unsorted_images.append(image)
                unsorted_masks.append(blank_output)

        unsorted_images = torch.stack(unsorted_images,dim=0)
        unsorted_masks = torch.stack(unsorted_masks,dim=0)
        # unsorted_images = unsorted_images.squeeze(1)
        # unsorted_masks = unsorted_masks.squeeze(1)
        print(f'unsorted images shape: {unsorted_images.shape}')
        print(f'unsorted masks shape: {unsorted_masks.shape}')

        std_vals_list = torch.clamp(torch.tensor(avg_std_dev),0.0,1.0)
        for i in range(len(std_vals_list)-1):
            if std_vals_list[i] >= 0.009:
                unconfident_images.append(unsorted_images[i])
                unconfident_masks.append(unsorted_masks[i])
            else: 
                confident_images.append(unsorted_images[i])
                confident_masks.append(unsorted_masks[i])

        confident_images = torch.cat(confident_images,dim=0)
        confident_masks = torch.cat(confident_masks,dim=0)
        unconfident_images = torch.cat(unconfident_images,dim=0)
        unconfident_masks = torch.cat(unconfident_masks,dim=0)
        print(f'unconfident images shape: {unconfident_images.shape}')
        print(f'unconfident masks shape: {unconfident_masks.shape}')
        print(f'unsorted images shape: {unsorted_images.shape}')
        print(f'unsorted masks shape: {unsorted_masks.shape}')
        total_dataset = torch.cat([unsorted_images,unsorted_masks],dim=1)
        confident_dataset = torch.cat([confident_images,confident_masks],dim=1)
        unconfident_dataset = torch.cat([unconfident_images,unconfident_masks],dim=1)

        return total_dataset,confident_dataset,unconfident_dataset 
        
        
                








def verification_visualization(total_dataset,confident_data,unconfident_data,save_path):

    confident_data = confident_data 
    status = int(input('Select menu option: 1 for uncertain review 2 for total review'))
    if status == 1: 
        for i in range(len(unconfident_data)-1):
            fig, axes = plt.subplots(1,2,figsize=(15,15))
            print(f'unconfident data shape: {unconfident_data.shape}')
            print(f'attempted image shape: {unconfident_data[i,0].shape}')
            print(f'attempted image range: {np.min(unconfident_data[i,1].detach().cpu().numpy())}/{np.max(unconfident_data[i,0].detach().cpu().numpy())}')
            axes[0].imshow(unconfident_data[i,0].detach().cpu().numpy(),cmap='gray')
            axes[1].imshow(threshold_image(unconfident_data[i,1].detach().cpu().numpy(),0.5),cmap='cividis')
            plt.savefig(os.path.join(save_path,'total_dataset_check.png'))
            plt.show() 
            valid = int(input('\nENTER 1 IF PREDICTION IS SATISFACTORY AND 0 OTHERWISE: '))
            if valid == 1: 
                print(f'confident_data shape: {confident_data.shape}')
                print(f'attempted add to confident data: {unconfident_data[i].shape}')
                confident_data = torch.cat([confident_data,unconfident_data[i].unsqueeze(0)],dim=0)
                unconfident_data = torch.cat([unconfident_data[:i],unconfident_data[i+1:]])


    if status == 2: 
        for i in range(len(total_dataset)-1):
            fig, axes = plt.subplots(1,2,figsize=(15,15))
            axes[0].imshow(total_dataset[i,0].detach().cpu().numpy(),cmap='gray')
            axes[1].imshow(total_dataset[i,1].detach().cpu().numpy(),cmap='cividis')
            plt.imsave(os.path.join(save_path,'total_dataset_check.png'))
            plt.show() 

    else: 
        print('value not recognized')

    return confident_data.detach().cpu().numpy(),unconfident_data.detach().cpu().numpy()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
detection_model = create_resnet18()
segmentation_model = UNet()
images_path = check_if_trailing_slash(IMAGES_PATH)
save_path = check_if_trailing_slash(SAVE_PATH)
train_images = generate_prediction_dataset()
image_loader = prediction_loaders(train_images)
save_path = '/home/henry/UCSF_Prostate_Segmentation/Data_plots/Sanity_checks'
total_data,confident_data,unconfident_data = inference(segmentation_model,segmentation_weights,
                                                       detection_model,detection_weights,
                                                       image_loader,device,clear_mem=False)

confident_data,unconfident_data = verification_visualization(total_data,confident_data,unconfident_data,save_path)

