import torch 
import numpy as np 
import matplotlib.pyplot as plt 
from Metrics.losses import IoULoss
import pandas as pd 
import os 
from tqdm import tqdm 
import time 

leaderboard_path = '/home/henry/UCSF_Prostate_Segmentation/Metrics/model_leaderboard.csv'
save_image_path = '/home/henry/UCSF_Prostate_Segmentation/Data_plots/leaderboards/'

def testing_loop(model_name,model,test_loader,device,num_epochs,clear_mem): 

    criterion = IoULoss()
    test_losses = []

    model = model.to(device)
    torch.cuda.empty_cache()
    with torch.no_grad():
        start_time = time.time()
        iters = 0 
        for i, batch in tqdm(enumerate(test_loader),total_len=(test_loader),desc ='Testing Model'): 
            img = batch[0].float.to(device)
            msk = batch[1].float.to(device)
            output = model(img)
            loss = criterion(output,msk)
            test_losses.append()
            iters += 1 
        end_time = time.time()
        total_time = ((end_time - start_time)/(iters * 2))

    test_iou_loss = float(sum(test_losses)/(len(test_losses)))

    return test_iou_loss, total_time 


def update_leaderboard(model_name,num_epochs,loss_function,lr, scheduler_name,iou_loss,time): 

    leaderboard_stats = {'Model':f'{model_name}','IoU Loss':f'{iou_loss}','Inference Time':{time},'Loss Function':f'{loss_function}',
                         'Epochs':f'{num_epochs}','Learning Rate':f'{lr}','Scheduler':f'{scheduler_name}'}
    leaderboard_stats_df = pd.DataFrame([leaderboard_stats])
    if os.path.exists(leaderboard_path): 
        existing_leaderboard = pd.read_csv(leaderboard_path)
    updated_leaderboard = pd.concat([existing_leaderboard,leaderboard_stats_df],ignore_index= True)
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

    axes.xaxis.set_visible(False)
    axes.yaxis.set_visible(False)
    axes.set_frame_on(False)

    table = axes.table(cellText=table_data.values,colLabels=table_data.columns,cellLoc='center',loc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2,1.2)

    plt.savefig(f'{save_image_path}model_leaderboard.png',bbox_inches='tight',pad_inches=0.1)
    plt.close()


def run_training(model_name,model,test_loader,device,num_epochs,
                 clear_mem,loss_function,lr,scheduler_name): 

    loss,time = testing_loop(model_name,model,test_loader,device,num_epochs,clear_mem)
    update_leaderboard(model_name,num_epochs,loss_function,lr,scheduler_name,loss,time)
    print(f'Test IoU: {loss}  Inference Speed: {time}/image')





    












