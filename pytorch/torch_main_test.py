import numpy as np 
import pandas as pd 
# import cv2 
import torch 
import torchvision
from torchvision import utils as vutils 
from torchvision import transforms 
from torch.nn import functional as F 
from torch.utils import data 
from torch.optim import SGD, Adam 
from torch_metrics import DiceLoss
from PIL import Image 
import matplotlib.pyplot as plt 
import os 
from statistics import mean 
# from architectures.torch_r2udense import r2udensenet
from architectures.torch_unet import UNet
from torch_data import load_train_data, load_test_data
from torch_plots import save_plots
import sklearn 
from torchvision.transforms import v2
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import datetime
import random 
from tqdm import tqdm
 



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
#densenet_weights_path = '/home/henry/UCSF_Prostate_Segmentation/densenet_weights'


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
def positives_only(images_train,mask_train): 
    num_positive = 0 
    num_negative = 0 
    positive_tumor_images = []
    positive_tumor_masks = []
    negative_tumor_images = []
    negative_tumor_masks = []
    for i, mask in enumerate(mask_train): 
        j = np.max(mask_train[i])
        if j > 0: 
            num_positive += 1
            positive_tumor_images.append(images_train[i])
            positive_tumor_masks.append(mask_train[i])
        else: 
            num_negative +=1
            negative_tumor_images.append(images_train[i])
            negative_tumor_masks.append(mask_train[i])
    images_train = positive_tumor_images
    mask_train = positive_tumor_masks
    return images_train,mask_train 


def sanity_check(images,masks): 

    fig, axes = plt.subplots(nrows=4,ncols=2,figsize=(15,15))
    for i in range(0,4): 
        rand = np.random.randint(0,len(images)-1)
        axes[i,0].imshow(images[rand],cmap='gray')
        axes[i,1].imshow(masks[rand],cmap='gray')
    plt.savefig("/data/ernesto/UCSF_Prostate_Segmentation/pytorch/test1")

def directories():
   log_directory_main ='.log'
   log_directory_test = os.path.join(log_directory_main + "/Test")
   
   if not os.path.exists(log_directory_main):
    print("DOESN'T EXIST")
    os.makedirs(log_directory_main)
    print(f'Directory created: {log_directory_main}')    
   else:
    print("Directory Exist")
   if not os.path.exists(log_directory_test):
    print("DOESN'T EXIST")
    os.makedirs(log_directory_test)
    print(f'Directory created: {log_directory_test}') 
   else:
    print("Directory Exist") 
 
    return log_directory_main,log_directory_test

def dataset_visualization(images,masks): 
    cont_bool = True 
    counter = 0 
    while cont_bool == True and counter < len(images): 
        print(f'Training Example #{counter}')
        fig, axes = plt.subplots(2,1,figsize=(15,15))
        axes[0].imshow(images[counter],cmap='gray')
        axes[1].imshow(masks[counter],cmap='gray')
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

class torch_loader(data.Dataset): 

    def __init__(self,inputs,transform=None): 
        self.inputs = inputs
        self.transform = transform 
        self.input_dtype = torch.float32
        self.target_dtype = torch.float32

    def __len__(self): 
        return len(self.inputs)
    
    def __getitem__(self,index):
        # torch.cuda.empty_cache() 
        image_array, mask_array = self.inputs[0]
        x = torch.from_numpy(np.transpose((np.array(image_array)),(2,0,1))).type(self.input_dtype)
        y = torch.from_numpy(np.transpose((np.array(mask_array)),(2,0,1))).type(self.target_dtype)
        # print(x.shape)
        print
        if self.transform is not None: 
            x = self.transform(x)
            y = self.transform(y)

        return x, y

#creates training and datasets in torch tensor form
#check that this returns the same datatype and the same shape as the one that worked with the other network 
def generate_dataset(positive_bool,augmentation_bool, augmentation_prob,val_size): 

    # images_train_tensor = []
    # mask_train_tensor = []
    # images_test_tensor = []
    # mask_test_tensor = []
    # dtype = torch.float64

    images_train, mask_train = load_train_data()
    images_test, mask_test = load_test_data()

    print(images_train.shape)
    print(mask_train.shape)    
    print(images_test.shape)
    print(mask_test.shape)

    images_train = normalization(images_train)
    images_test = normalization(images_test)
    mask_train = mask_train.astype(np.float32) / 255.
    mask_test = mask_test.astype(np.float32) / 255.

    print(images_train.shape)
    print(mask_train.shape)    
    print(images_test.shape)
    print(mask_test.shape)

    if positive_bool: 
        images_train, mask_train = positives_only(images_train,mask_train)
        images_test,mask_test = positives_only(images_test,mask_test)
        
    if augmentation_bool: 
        for i in range(0,len(images_train)-1): 
            prob = random.random()
            if prob < augmentation_prob: 
                im = images_train[i]
                mask = mask_train[i]
                aug_im, aug_mask = random_augmentation(im,mask)
                images_train[i] = aug_im
                mask_train[i] = aug_mask
            else: 
                pass 

    mask_test = np.expand_dims(np.array(mask_test),axis=-1)

    train_loader_data = [(images_train[i],mask_train[i])for i in range(len(images_train)-1)]
    test_loader_data = [(images_test[i],mask_test[i])for i in range(len(images_test)-1)]
    # print(train_loader_data.shape)
    # print(test_loader_data.shape)
    train_data = torch_loader(train_loader_data)
    test_data = torch_loader(test_loader_data)

    return train_data,test_data
 


    #transform = torchvision.transforms.

    # for i in range(0,len(images_train)-1):
    #     im = images_train[i]
    #     im = torch.from_numpy(np.resize(im, (1,128,128))).type(dtype)       
    #     images_train_tensor.append(im)
    #     mask = mask_train[i]
    #     mask = torch.from_numpy(np.resize(mask, (1,128,128))).type(dtype)
    #     mask_train_tensor.append(mask)

    # for i in range(0,len(images_test)-1):
    #     im = images_test[i]
    #     im = torch.from_numpy(np.resize(im, (1,128,128))).type(dtype)       
    #     images_test_tensor.append(im)
    #     mask = mask_test[i]
    #     mask = torch.from_numpy(np.resize(mask, (1,128,128))).type(dtype)
    #     mask_test_tensor.append(mask)
        
    # images_train_tensor = torch.stack(images_train_tensor)
    # mask_train_tensor = torch.stack(mask_train_tensor)
    # images_test_tensor = torch.stack(images_test_tensor)
    # mask_test_tensor = torch.stack(mask_test_tensor)

    # print(images_train_tensor.shape)
    # print(mask_train_tensor.shape)

    # train_dataset = data.TensorDataset(images_train_tensor,mask_train_tensor)
    # test_dataset = data.TensorDataset(images_test_tensor,mask_test_tensor)

    # return train_dataset, test_dataset

#defining train and val loaders 
def loaders(train_dataset, val_amount,batch_size): 
    validation_length = int(val_amount * len(train_dataset))
    remainder = validation_length % batch_size
    if remainder != 0: 
        validation_length - remainder
    train_set,val_set = data.random_split(train_dataset,[len(train_dataset)-validation_length,validation_length])

    train_loader = data.DataLoader(dataset=train_set,batch_size=batch_size,shuffle=True)
    val_loader = data.DataLoader(dataset=val_set,batch_size=batch_size,shuffle=False)

    return train_loader,val_loader

   #  validation_length = int(val_amount * len(train_dataset))
    # train_set,val_set = data.random_split(train_dataset,[len(train_dataset)-validation_length,validation_length])

    # train_loader = data.DataLoader(dataset=train_set,batch_size=batch_size,shuffle=True)
    # val_loader = data.DataLoader(dataset=val_set,batch_size=batch_size,shuffle=False)

    # return train_loader,val_loader


def train(model_name, model, optimizer, criterion, train_loader, val_loader, device, num_epochs, clear_mem):
    torch.cuda.empty_cache() 
    print(f"Using device: {device}")
    print(f'Model sent to {device}')
    model = model.to(device)
    all_dice_train_losses = []
    all_dice_val_losses = []
    iters = 0

    for epoch in range(num_epochs): 
        print(f"Epoch {epoch+1} / {num_epochs}")
        
        model.train()    
        dice_train_losses = []

        for i, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Training Epoch:{epoch+1}/{num_epochs}"): 
            try:
                img = batch[0].float().to(device)
                msk = batch[1].float().to(device)
                optimizer.zero_grad()
                output = model(img)
                loss = criterion(output, msk)
                loss.backward()
                optimizer.step()
                
                dice_train_losses.append(loss.item())
                iters += 1
            except Exception as e:
                print(f"Error during training at iteration {i}: {e}")
        
        all_dice_train_losses.append(sum(dice_train_losses) / len(dice_train_losses))

        model.eval()
        dice_val_losses = []
        dice_val_preds = []
        dice_val_labels = []
        
        with torch.no_grad():
            for i, batch in enumerate(val_loader): 
                try:
                    torch.cuda.empty_cache()  
                    img = batch[0].float().to(device)
                    msk = batch[1].float().to(device)
                    output = model(img)
                    loss = criterion(output, msk)
                    dice_val_losses.append(loss.item())
                    dice_val_labels.append(msk)
                    dice_val_preds.append(output)
                except Exception as e:
                    print(f"Error during validation at iteration {i}: {e}")
        
        all_dice_val_losses.append(sum(dice_val_losses) / len(dice_val_losses))
        print(f'Epoch {epoch+1} completed. Train Loss: {all_dice_train_losses[-1]}, Val Loss: {all_dice_val_losses[-1]}')

        plot_dir = log_directory_test + "/plots"
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        
        save_plots(
            all_dice_train_losses,
            all_dice_val_losses,
            plot_dir
        )
       
    results = {
            'model_name': model_name,
            'train_losses': all_dice_train_losses,
            'val_losses': all_dice_val_losses,
        }
    print(results)
        # torch.cuda.empty_cache() 
        
    
    #     plot_dir = log_directory_test + "/plots"
    #     print(plot_dir)
    #     if not os.path.exists(plot_dir):
    #          os.makedirs(plot_dir)
    #          print("create")   
    #          save_plots(
    #          all_dice_train_losses,
    #          all_dice_val_losses,
    #          plot_dir
    #          )
       
    # results = {
    #         'model_name': model_name,
    #         'train_losses': all_dice_train_losses,
    #         'val_losses': all_dice_val_losses,
    #         }
    # print(results)
        
    if clear_mem:
        del model, optimizer, criterion
        torch.cuda.empty_cache()    
    return results
    # results = {
    # 'model_name' : model_name,
    # 'train_losses': all_dice_train_losses,
    #     # 'train_scores': all_train_scores,
    # 'val_losses': all_dice_val_losses,
    #     # 'val_scores': val_scores,
    # }
    # print(results)

    # save_path = save_model_weights_path("/data/ernesto/UCSF_Prostate_Segmentation/pytorch",model_name)
    # torch.save(model.state_dict(),save_path)

    # if clear_mem: 
    #     del model,optimizer,criterion
    #     torch.cuda.empty_cache()
   
    
    # return results 


def visualize_segmentation(model,data_loader,num_samples=5,device='cuda'): 
    fig, axes = plt.subplots(num_samples,3,figsize=(15,15))
    num_samples_count = 0 
    for ax, col in zip(axes[0],['MRI','Ground Truth','Predicted Mask']): 
        ax.set_title(col)
    index=0
    model.eval()
    for i, batch in enumerate(data_loader): 
        print(i)
        img = batch[0].float()
        img = img.to(device)
        msk = batch[1].float()
        msk = msk.to(device)
        output = model(img)
        if i % 15 == 0: 
            axes[num_samples_count,0].imshow(torch.squeeze(img[0],dim=0).detach().cpu().numpy(),
                            cmap='gray',interpolation='none')
            axes[num_samples_count,1].imshow(torch.squeeze(msk[0],dim=0).detach().cpu().numpy(),
                            cmap='gray',interpolation='none')
            axes[num_samples_count,2].imshow(torch.squeeze(output[0],dim=0).detach().cpu().numpy(),
                            cmap='gray',interpolation='none')
            num_samples_count += 1
        if num_samples_count >= (num_samples)-1:
            break

    plt.tight_layout()
    plt.savefig("/data/ernesto/UCSF_Prostate_Segmentation/pytorch/Test1.png")

if __name__ == '__main__':
 log_directory_main,log_directory_test= directories()
 train_set,test_set = generate_dataset(positive_bool=False,augmentation_bool=False,
                                      augmentation_prob=None,val_size=0.1)
 train_loader,val_loader = loaders(train_set,0.2,batch_size=2)

 model_name = "test"
 model = UNet()
#  spec_loss = torch.nn.BCEWithLogitsLoss()
 optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
 criterion = DiceLoss()
 device = torch.device('cuda'if torch.cuda.is_available() else "cpu")
 num_epochs = 2
 results = train(model_name,model,optimizer,criterion,
                train_loader,val_loader,device,
                num_epochs,clear_mem=True)

visualize_segmentation(model,val_loader,num_samples=5,device='cuda')




    

    

    

    
    
    

    


























