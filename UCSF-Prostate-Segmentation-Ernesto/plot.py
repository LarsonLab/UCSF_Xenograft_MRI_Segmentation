import os
import matplotlib.pyplot as plt

def save_plots(fold_num,train_comp_loss,val_comp_loss,train_dice_coef,val_dice_coef,train_dice_loss,val_dice_loss,train_iou,val_iou,out_dir):
 plt.style.use('ggplot')

#COMP Plot
 plt.title(f'Model #{fold_num}')
 plt.figure(figsize =(10,7))
 plt.plot(
 train_comp_loss,
 color='tab:blue',
 linestyle = '-',
 label = 'train composite loss')
 plt.plot(
 val_comp_loss,
 color='tab:red',
 linestyle='-', 
 label='validataion composite loss')
 plt.xlabel('Epochs')
 plt.ylabel('Composite Loss')
 plt.legend()
 plt.savefig(f'{out_dir}/Composite_Loss_{fold_num}.png')


#DICE COEF
 plt.title(f'Model #{fold_num}')
 plt.figure(figsize =(10,7))
 plt.plot(
 train_dice_coef,
 color='tab:blue',
 linestyle = '-',
 label = 'train dice coef')
 plt.plot(
 val_dice_coef,
 color='tab:red',
 linestyle='-', 
 label='validataion dice coef')
 plt.xlabel('Epochs')
 plt.ylabel('Dice Coef')
 plt.legend()
 plt.savefig(f'{out_dir}/Dice_Coef_{fold_num}.png')

 

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
 label='validataion dice loss')
 plt.xlabel('Epochs')
 plt.ylabel('Dice Loss')
 plt.legend()
 plt.savefig(f'{out_dir}/Dice_Loss_{fold_num}.png')

#IOU Loss
 plt.title(f'Model #{fold_num}')
 plt.figure(figsize =(10,7))
 plt.plot(
 train_iou,
 color='tab:blue',
 linestyle = '-',
 label = 'train iou')
 plt.plot(
 val_iou,
 color='tab:red',
 linestyle='-', 
 label='validataion iou')
 plt.xlabel('Epochs')
 plt.ylabel('Dice Loss')
 plt.legend()
 plt.savefig(f'{out_dir}/IoU_{fold_num}.png')
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

 
