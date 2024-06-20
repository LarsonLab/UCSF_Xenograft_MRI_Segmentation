import os
import matplotlib.pyplot as plt

def save_plots(all_train_dice_loss,all_val_dice_loss,out_dir):
 plt.style.use('ggplot')

 #DICE LOSS
 plt.title(f'Dice Loss')
 plt.figure(figsize =(10,7))
 plt.plot(
 all_train_dice_loss,
 color='tab:blue',
 linestyle = '-',
 label = 'train dice loss')
 plt.plot(
 all_val_dice_loss,
 color='tab:red',
 linestyle='-', 
 label='validation dice loss')
 plt.xlabel('Epochs')
 plt.ylabel('Dice Loss')
 plt.legend()
 plt.savefig(f'{out_dir}/Dice_Loss.png')