import torch 
import numpy as np 
import matplotlib.pyplot as plt 
from Metrics.losses import IoULoss
import pandas as pd 
import os 
from tqdm import tqdm 
import time 
from architectures.Attention_UNet import Attention_UNet
from architectures.torch_unet import UNet
from architectures.Mamba_Unet import LightMUNet

leaderboard_path = '/home/henry/UCSF_Prostate_Segmentation/Metrics/model_leaderboard.csv'
save_image_path = '/home/henry/UCSF_Prostate_Segmentation/Data_plots/leaderboards/'

def testing_loop(model_name,model,weights_pth,test_loader,device,num_epochs,clear_mem): 

    criterion = IoULoss()
    test_losses = []
    print(weights_pth)
    model.load_state_dict(torch.load(weights_pth))
    model = model.to(device)
    model.eval()
    torch.cuda.empty_cache()
    with torch.no_grad():
        start_time = time.time()
        iters = 0 
        for i, batch in tqdm(enumerate(test_loader),total=len(test_loader),desc =f'Testing {model_name} Model'): 
            img = batch[0].float().to(device)
            msk = batch[1].float().to(device)
            output = model(img)
            loss = criterion(output,msk)
            test_losses.append(loss.item())
            iters += 1 
        end_time = time.time()
        total_time = ((end_time - start_time)/(iters * 2))

    test_iou_loss = float(sum(test_losses)/(len(test_losses)))
    print(f'Test IoU Loss: {test_iou_loss:.4f}')

    return test_iou_loss, total_time 


def update_leaderboard(model_name,num_epochs,loss_function,lr, scheduler_name,iou_loss,time): 

    loss_score = 1.0 - iou_loss

    leaderboard_stats = {'Model':f'{model_name}','IoU Loss':float(f'{loss_score:.4f}'),'Inference Time':f'{time:.4f}s/image','Loss Function':f'{loss_function}',
                         'Epochs':f'{num_epochs}','Learning Rate':f'{lr}','Scheduler':f'{scheduler_name}'}
    leaderboard_stats_df = pd.DataFrame([leaderboard_stats])
    if os.path.exists(leaderboard_path): 
        try:
            existing_leaderboard = pd.read_csv(leaderboard_path)
            updated_leaderboard = pd.concat([existing_leaderboard,leaderboard_stats_df],ignore_index= True)
        except pd.errors.EmptyDataError:
            updated_leaderboard = leaderboard_stats_df
    else: 
        updated_leaderboard = leaderboard_stats_df
    updated_leaderboard = updated_leaderboard.sort_values(by=['IoU Loss'],ascending=False)
    plot_leaderboard(updated_leaderboard)
    updated_leaderboard.to_csv(leaderboard_path,index=False)



def plot_leaderboard(dataframe): 

        
    if len(dataframe) <= 5: 
        table_data = dataframe 
        fig, axes = plt.subplots(nrows=len(dataframe),ncols=6)
    elif len(dataframe) <= 10: 
        table_data = dataframe.head(5)
        fig, axes = plt.subplots(nrows=5,ncols=6) 
    else: 
        table_data = dataframe.head(10)
        fig, axes = plt.subplots(nrows=10,ncols=6)

    fig, ax = plt.subplots(figsize=(12,2 + len(table_data)*0.4))

    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_frame_on(False)


    table = ax.table(cellText=table_data.values,colLabels=table_data.columns,cellLoc='center',loc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2,1.2)

    plt.savefig(f'{save_image_path}model_leaderboard.png',bbox_inches='tight',pad_inches=0.1)
    plt.close()


def run_testing(model_name,model,weights_path,test_loader,device,num_epochs,
                 clear_mem,loss_function,lr,scheduler_name): 

    # if num_epochs == 100:
    #     loss,time = testing_loop(model_name,model,weights_path,test_loader,device,num_epochs,clear_mem)
    #     update_leaderboard(model_name,num_epochs,loss_function,lr,scheduler_name,loss,time)
    # else: 
    #     print('Testing only conducted on 100 epoch runs')

    loss,time = testing_loop(model_name,model,weights_path,test_loader,device,num_epochs,clear_mem)
    print(f'Test IoU: {loss}  Inference Speed: {time}/image')
    update_leaderboard(model_name,num_epochs,loss_function,lr,scheduler_name,loss,time)
    


def model_selection(model_name: str):
    if model_name == 'Attention':
        model = Attention_UNet()
    if model_name == 'Vanilla':
        model = UNet()
    if model_name == 'Mamba': 
        model = LightMUNet()
        






    












