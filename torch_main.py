import numpy as np 
import pandas as pd 
import cv2 
import torch 
import torchvision
from torchvision import utils as vutils 
from torchvision import transforms 
from torch.nn import functional as F 
from torch.utils import data 
from torch.optim import SGD, Adam 
from torch_metrics import dice_loss, BCE, BCE_dice_loss, BCEWithLogitsLoss, IoU
from PIL import Image 
import matplotlib.pyplot as plt 
import os 
from statistics import mean 
from architectures.torch_r2udense import r2udensenet
from architectures.torch_unet import UNet
from data2D_ucsf_1d import load_train_data, load_test_data
import sklearn 
from torchvision.transforms import v2
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import datetime
import random 
import random 



#sanity check metrics and directories 
file_no_mask = 0 
maskless_files = []
mask_no_file = 0 
fileless_masks = []

num_negative_diagnoses = 0
num_positive_diagnoses = 0 

img_dimensions = []
msk_dimensions = []

n_files = 0 
densenet_weights_path = '/home/henry/UCSF_Prostate_Segmentation/densenet_weights'


#metrics 
composite_loss = []
dice_loss_list = []
dice = []
precison = []
recall = []
auc = []
accuracy = []
IoU_test = []
binaryCE_original = []
bce_list = []

#change these functions to list the name of whichever model we choose to use 
#model name should be a variable 
def get_model_name(k,e): 
    return 'model_r2udensenet'+str(k)+"_"+str(e)+'.hdf5'
def get_log_name(k): 
    return 'log_r2udensenet'+str(k)+'.csv'

def save_model_weights_path (path,model_name): 
    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_path = os.path.join(path,f'{current_time}_#_{model_name}.pth')
    return save_path


#random image augmentation 
def random_augmentation(image,mask): 
    transforms = v2.Compose([
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomVerticalFlip(p=0.5),
    v2.RandomRotation(degrees=(-15,15))
])
    image = transforms(image)
    mask = transforms(mask)
    return image,mask

#filter out negative diagnoses
def positives_only(x_train,y_train): 
    num_positive = 0 
    num_negative = 0 
    positive_tumor_images = []
    positive_tumor_masks = []
    negative_tumor_images = []
    negative_tumor_masks = []
    for i, mask in enumerate(y_train): 
        j = np.max(y_train[i])
        if j > 0: 
            num_positive += 1
            positive_tumor_images.append(x_train[i])
            positive_tumor_masks.append(y_train[i])
        else: 
            num_negative +=1
            negative_tumor_images.append(x_train[i])
            negative_tumor_masks.append(y_train[i])
    x_train = positive_tumor_images
    y_train = positive_tumor_masks
    return x_train,y_train 


def sanity_check(images,masks): 

    fig, axes = plt.subplots(nrows=4,ncols=2,figsize=(15,15))
    for i in range(0,4): 
        rand = np.random.randint(0,len(images)-1)
        axes[i,0].imshow(images[rand],cmap='gray')
        axes[i,1].imshow(masks[rand],cmap='gray')

    plt.show()

def dataset_visualization(images,masks): 
    cont_bool = True 
    counter = 0 
    while cont_bool == True and counter < len(images): 
        fig, axes = plt.subplots(2,1,figsize=(15,15))
        axes[0].imshow(images[counter],cmap='gray')
        axes[1].imshow(masks[counter],cmap='gray')
        plt.set_title(f'Training example #{counter}')
        plt.show()
        cont_state = int(input('Type 1 to continue or 0 to exit: '))
        if cont_state == 1: 
            counter += 1
            continue 
        else:
            cont_state = False 

def normalization(data): 
    data_mean = np.mean(data)
    data_std = np.std(data)
    data_normalized = (data - data_mean)/data_std
    return data_normalized




#creates training and datasets in torch tensor form
#check that this returns the same datatype and the same shape as the one that worked with the other network 
def generate_dataset(positive_bool,augmentation_bool, augmentation_prob,val_size): 

    x_train_tensor = []
    y_train_tensor = []
    x_test_tensor = []
    y_test_tensor = []
    dtype = torch.float64

    x_train, y_train = load_train_data()
    x_test, y_test = load_test_data()

    x_train= normalization(x_train)
    x_test = normalization(x_test)
    y_train = y_train.astype(np.float32) / 255.
    y_test = y_test.astype(np.float32) / 255.

    if positive_bool: 
        x_train, y_train = positives_only(x_train,y_train)
        x_test,y_test = positives_only(x_test,y_test)
        
    if augmentation_bool: 
        for i in range(0,len(x_train)-1): 
            prob = random.random()
            if prob < augmentation_prob: 
                im = x_train[i]
                mask = y_train[i]
                aug_im, aug_mask = random_augmentation(im,mask)
                x_train[i] = aug_im
                y_train[i] = aug_mask
            else: 
                pass 
    
    sanity_check(x_train,y_train)


    #transform = torchvision.transforms.

    for i in range(0,len(x_train)-1):
        im = x_train[i]
        im = torch.from_numpy(np.resize(im, (1,128,128))).type(dtype)       
        x_train_tensor.append(im)
        mask = y_train[i]
        mask = torch.from_numpy(np.resize(mask, (1,128,128))).type(dtype)
        y_train_tensor.append(mask)

    for i in range(0,len(x_test)-1):
        im = x_test[i]
        im = torch.from_numpy(np.resize(im, (1,128,128))).type(dtype)       
        x_test_tensor.append(im)
        mask = y_test[i]
        mask = torch.from_numpy(np.resize(mask, (1,128,128))).type(dtype)
        y_test_tensor.append(mask)
        
    x_train_tensor = torch.stack(x_train_tensor)
    y_train_tensor = torch.stack(y_train_tensor)
    x_test_tensor = torch.stack(x_test_tensor)
    y_test_tensor = torch.stack(y_test_tensor)

    print(x_train_tensor.shape)
    print(y_train_tensor.shape)

    train_dataset = data.TensorDataset(x_train_tensor,y_train_tensor)
    test_dataset = data.TensorDataset(x_test_tensor,y_test_tensor)

    return train_dataset, test_dataset

#defining train and val loaders 
def train_val_loader(train_dataset, val_amount,batch_size): 
    validation_length = int(val_amount * len(train_dataset))
    train_set,val_set = data.random_split(train_dataset,[len(train_dataset)-validation_length,validation_length])

    train_loader = data.DataLoader(dataset=train_set,batch_size=batch_size,shuffle=True)
    val_loader = data.DataLoader(dataset=val_set,batch_size=batch_size,shuffle=False)

    return train_loader,val_loader



def train(model_name,model,optimizer,criterion,spec_loss,train_loader,val_loader,device,num_epochs,clear_mem=True):


    torch.cuda.empty_cache() 
    print('Model sent to '+str(device))
    model.to(device)
    losses=[]
    train_scores = []
    iters = 0
    for epoch in range(num_epochs): 
        if epoch %5 == 0: 
            print(f"Epoch {epoch+1} / {num_epochs}")
        for i, batch in enumerate(train_loader): 
            img = batch[0].float()
            img = img.to(device)
            msk = batch[1].float()
            msk = msk.to(device)
            optimizer.zero_grad()
            output = model(img)
            loss = criterion(output,msk)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            #need to fill this in with the loss function
            train_scores.append(spec_loss(output.detach(),msk))
            iters += 1

    model.eval()
    val_losses = []
    val_scores = []
    val_preds = []
    val_labels = []

    for i, batch in enumerate(val_loader): 
        img = batch[0].float()
        img = img.to(device)
        msk = batch[1].float()
        msk = msk.to(device)
        val_labels.append(msk)
        output = model(img)
        val_preds.append(output)
        loss = criterion(output,msk)
        #need to fill this in with the loss function 
        val_scores.append(spec_loss(output.detach(),msk))
        val_scores.append(loss.item())

    auroc = roc_auc_score(val_labels.cpu().numpy(),val_preds.cpu().numpy()[:,1])

    results = {
        'model_name' : model_name,
        'train_scores': losses,
        'train_scores': train_scores,
        'val_losses': val_losses,
        'val_scores': val_scores, 
        'roc_auc_score': auroc

    }
    save_path = save_model_weights_path(densenet_weights_path,f'{num_epochs}')
    torch.save(model.state_dict(),save_path)

    if clear_mem: 
        del model 
        del optimizer 
        del criterion 
        torch.cuda.empty_cache()

    return results 


def visualize_segmentation(model,data_loader,num_samples=5,device='cuda'):
    fig, axs = plt.subplots(nrows=num_samples,ncols=3,figsize=(60,60))
    for ax, col in zip(axs[0],['MRI','Ground Truth',
                               'Predicted Mask']):
        ax.set_title(col)
    index=0
    for i,batch in enumerate(data_loader): 
        img = batch[0].float()
        img = img.to(device)
        msk = batch[1].float()
        msk = msk.to(device)
        output = model(img)

        for j in range(batch[0].size()[0]):
            axs[index,0].imshow(np.transpose(img[j].detach().cpu().numpy(),
                (1,2,0)).astype(np.uint8),cmap='bone',interpolation='none')
            axs[index,1].imshow(np.transpose(img[j].detach().cpu().numpy(),
                (1,2,0)).astype(np.uint8),cmap='bone',interpolation='none')
            axs[index,1].imshow(torch.squeeze(msk[j]).detach().cpu().numpy(),
                                cmap='Blues',interpolation='none',alpha=0.5)
            axs[index,2].imshow(np.transpose(img[j].detach().cpu().numpy(),
                (1,2,0)).astype(np.uint8),cmap='bone',interpolation='none')
            axs[index,2].imshow(torch.squeeze(output[j]).detach().cpu().numpy(),
                                cmap='Greens',interpolation='none',alpha=0.5)
            
            index += 1
        
        if index >= num_samples: 
            break

    plt.tight_layout()

train_set,test_set = generate_dataset(positive_bool=True,augmentation_bool=False,
                                      augmentation_prob=None,val_size=0.1)
train_loader,val_loader = train_val_loader(train_set,0.2,batch_size=2)


model = r2udensenet()
spec_loss = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
criterion = BCE_dice_loss
device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_epochs = 50 
results = train('run1',model,optimizer,criterion,
                spec_loss,train_loader,val_loader,device,
                num_epochs=num_epochs,clear_mem=True)

visualize_segmentation(model,val_loader,num_samples=5,device='cuda')




    

    

    

    
    
    

    


























