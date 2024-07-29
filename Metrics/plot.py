import os
import matplotlib.pyplot as plt
import datetime 
import csv 
import numpy as np 
import statsmodels.api as sm  


current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

def save_plots(fold_num,train_comp_loss,val_comp_loss,train_dice_coef,val_dice_coef,train_dice_loss,val_dice_loss,train_iou,val_iou,out_dir):
 plt.style.use('ggplot')

#COMP Plot
#  plt.title(f'Model #{fold_num}')
#  plt.figure(figsize =(10,7))
#  plt.plot(
#  train_comp_loss,
#  color='tab:blue',
#  linestyle = '-',
#  label = 'train composite loss')
#  plt.plot(
#  val_comp_loss,
#  color='tab:red',
#  linestyle='-', 
#  label='validation composite loss')
#  plt.xlabel('Epochs')
#  plt.ylabel('Composite Loss')
#  plt.legend()
#  plt.savefig(f'{out_dir}/Composite_Loss_{fold_num}.png')


#DICE COEF
#  plt.title(f'Model #{fold_num}')
#  plt.figure(figsize =(10,7))
#  plt.plot(
#  train_dice_coef,
#  color='tab:blue',
#  linestyle = '-',
#  label = 'train dice coef')
#  plt.plot(
#  val_dice_coef,
#  color='tab:red',
#  linestyle='-', 
#  label='validation dice coef')
#  plt.xlabel('Epochs')
#  plt.ylabel('Dice Coef')
#  plt.legend()
#  plt.savefig(f'{out_dir}/Dice_Coef_{fold_num}.png')

 

#DICE LOSS
 plt.title(f'Model #{fold_num}')
 plt.figure(figsize =(10,7))
 plt.plot(
 train_dice_loss,
 color='tab:blue',
 linestyle = '-',
 label = 'train dice loss')
 plt.plot(
 val_dice_loss,
 color='tab:red',
 linestyle='-', 
 label='validation dice loss')
 plt.xlabel('Epochs')
 plt.ylabel('Dice Loss')
 plt.legend()
 plt.savefig(f'{out_dir}/Dice_Loss_{fold_num}.png')

#IOU Loss
#  plt.title(f'Model #{fold_num}')
#  plt.figure(figsize =(10,7))
#  plt.plot(
#  train_iou,
#  color='tab:blue',
#  linestyle = '-',
#  label = 'train iou')
#  plt.plot(
#  val_iou,
#  color='tab:red',
#  linestyle='-', 
#  label='validation iou')
#  plt.xlabel('Epochs')
#  plt.ylabel('Dice Loss')
#  plt.legend()
#  plt.savefig(f'{out_dir}/IoU_{fold_num}.png')
# # Train/Valid Dice Loss plots.
#  plt.figure(figsize=(10, 7))
#  plt.plot(
#  train_dice,
#  color='tab:red',
#  linestyle = '-',
#  label='train dice loss'
#  )
 
#  plt.xlabel('Epochs')
#  plt.ylabel('Dice Loss')
#  plt.legend()
#  plt.savefig(f'{out_dir}/dice.png')

# #Train/Valid Mious plots
#  plt.figure(figsize=(10, 7))
#  plt.plot(
#  train_miou,
#  color='tab:blue',
#  linestyle = '-',
#  label='train IoU'
#  )
#  plt.xlabel('Epochs')
#  plt.ylabel('train IoU')
#  plt.legend()
#  plt.savefig(f'{out_dir}/iou.png')


def save_plots2(opt_train_loss,opt_val_loss,train_comp_loss,val_comp_loss,train_dice_coef,val_dice_coef,train_iou,val_iou,save_dir):
    fig, axes = plt.subplots(nrows=4,ncols=2,figsize=(20,40))
    for ax, col in zip(axes[0],['Train','Validation']):
       ax.set_title(col)

    y_labels = ['Composite Loss', 'Dice Coefficient', 'IoU', 'Loss Used by Optimizer']
    x_label = 'Epochs' 

    axes[0,0].plot(train_comp_loss,color='tab:blue',linestyle='-',label = 'composite loss')
    axes[0,1].plot(val_comp_loss,color='tab:blue',linestyle='-',label = 'composite loss')
    axes[1,0].plot(train_dice_coef,color='tab:blue',linestyle='-',label = 'dice loss')
    axes[1,1].plot(val_dice_coef,color='tab:blue',linestyle='-',label = 'dice loss')
    axes[2,0].plot(train_iou,color='tab:blue',linestyle='-',label= 'iou loss')
    axes[2,1].plot(val_iou,color='tab:blue',linestyle='-',label= 'iou loss')
    axes[3,0].plot(opt_train_loss,color='tab:blue',linestyle='-',label= 'loss used by opt')
    axes[3,1].plot(opt_val_loss,color='tab:blue',linestyle='-',label= 'loss used by opt')

    for ax in axes.flatten():
        ax.legend() 

    for i in range(4):  # Iterate over rows
        for j in range(2):  # Iterate over columns
            axes[i,j].set_xlabel(x_label)
            axes[i,j].set_ylabel(y_labels[i])

    plt.tight_layout()
    plt.savefig(f'{save_dir}{current_time}.png')
    plt.close()

def save_plots3(all_train_dice_loss,all_val_dice_loss,all_val_ref_loss,loss_name,out_dir):
 plt.style.use('ggplot')

 #DICE LOSS
 plt.figure()
 plt.plot(all_train_dice_loss, label='Train Loss')
 plt.plot(all_val_dice_loss, label='Validation Loss')
 plt.plot(all_val_ref_loss,label='Reference Loss')
 plt.xlabel('Epochs')
 plt.ylabel('Loss')
 plt.legend()
 plt.title(f'Training and Validation {loss_name} Loss vs Reference')
 plot_path = os.path.join(out_dir, 'loss_plot.png')
 plt.savefig(f'{out_dir}{current_time}')
 plt.close()
 print(f"Plot saved at: {plot_path}")
 print('done plotting')


 def csv_to_list(file_name,run_name):
  
    desired_data = []
  
    with open(file_name,'r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            run_name_ex,epoch,loss = row[0],row[1],row[2]
            if run_name_ex == run_name:
               desired_data.append(float(loss))

        return desired_data
    
def csv_to_list2(file_name,run_name,loss_name):
   
    desired_data = []
    with open(file_name,'r') as file:
        reader = csv.reader(file)
        for row in reader:
            run_name_ex,loss_name_ex,epoch,loss = row[0],row[1],row[2], row[3]
            if run_name_ex == run_name:
                if loss_name_ex == loss_name:
                    desired_data.append(float(loss))

    return desired_data 

def create_loss_plot(loss1, loss1name,loss2, loss2name,loss3, loss3name,loss4,loss4name, model_name,save_path,comparison_metric):
    plt.style.use('ggplot')

    #DICE LOSS
    plt.figure()
    plt.plot(loss1, label=f'{loss1name}')
    plt.plot(loss2, label=f'{loss2name}')
    plt.plot(loss3,label=f'{loss3name}')
    plt.plot(loss4,label=f'{loss4name}')
    plt.xlabel('Epochs')
    plt.ylabel(f'{comparison_metric}')
    plt.legend()
    plt.title(f'Comparative Convergence in Terms of {comparison_metric}')
    save_path = os.path.join(save_path, f'{current_time}{model_name}.png')
    plt.savefig(f'{save_path}')
    plt.close()
    print(f"Plot saved at: {save_path}")
    print('done plotting')

def create_correlation_plot(run_name,std_dev,loss,loss_name,save_path):
   plt.scatter(std_dev,loss)
   plt.plot(np.unique(std_dev),np.poly1d(np.polyfit(std_dev,loss,1))(np.unique(std_dev)),color='red')
   plt.xlabel('Standard Deviation')
   plt.ylabel(f'{loss_name} Loss')
   plt.title(f'Correlation Between {loss_name} Loss and Prediction Certainty')
   plt.grid(True,linestyle='--',alpha=0.7)
   save_path = os.path.join(save_path,f'{current_time}{loss_name}.png')
   plt.savefig(f'{save_path}')
   plt.close()
   print(f'Certainty plot saved at {save_path}')
   print('done certainty plotting')


def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i,y[i],f'{y[i]}',ha='center')

def create_full_bar_plot(vals,xlab,ylab,title_custom, save_path):
    names = list(vals.keys())
    values = list(vals.values())
    fig = plt.figure(figsize = (6,8))

    plt.bar(names,values,color=('red','orange','blue','pink'),width=0.6)
    addlabels(names,values)
    plt.xlabel(f'{xlab}')
    plt.ylabel(f'{ylab}')
    plt.title(f'{title_custom}')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path,f'{title_custom}.png'))
    plt.close()

def create_full_bar_plot2(vals,xlab,ylab,title_custom, save_path):
    names = list(vals.keys())
    values = list(vals.values())
    fig = plt.figure(figsize = (20,8))

    plt.bar(names,values,color=('blue','red','blue','red','blue','red','blue','red'),width=0.9)
    addlabels(names,values)
    plt.xlabel(f'{xlab}')
    plt.ylabel(f'{ylab}')
    plt.title(f'{title_custom}')
    plt.tight_layout
    plt.savefig(os.path.join(save_path,f'{title_custom}.png'))
    plt.close()


def leverage_plot(x,y):

    print(len(x))
    sm.add_constant(x)
    model = sm.OLS(y,x).fit()
    influence = model.get_influence()
    leverage = influence.hat_matrix_diag
    num = (2*(1+1))/len(x)
    high_leverage_indices = np.where(leverage > num)[0]
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.axvline(x=num,color='red',linestyle='--',label='Leverage Boundary')
    sm.graphics.influence_plot(model, criterion="cooks", ax=ax)
    plt.savefig(f'/home/henry/UCSF_Prostate_Segmentation/Metrics/presentation_graphics/influence_plot{current_time}.png')
    plt.close()
    return high_leverage_indices


def create_multi_scatter_plot(x1,y1,x2,y2,title,loss_name):

    plt.scatter(x1,y1,c='blue',label='Under Leverage Threshold')
    plt.scatter(x2,y2,c='red',label='Over Leverage Threshold')
    plt.legend()
    plt.title(f'{title}')
    plt.xlabel('Prediction Certainty')
    plt.ylabel(f'{loss_name} loss')






