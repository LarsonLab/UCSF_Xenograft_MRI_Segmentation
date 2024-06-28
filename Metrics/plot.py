import os
import matplotlib.pyplot as plt
import datetime 

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
