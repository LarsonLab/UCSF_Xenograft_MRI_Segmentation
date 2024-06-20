import os
import matplotlib.pyplot as plt

def save_plots(all_train_dice_loss,all_val_dice_loss,out_dir):
 plt.style.use('ggplot')

 #DICE LOSS
 plt.figure()
 plt.plot(all_train_dice_loss, label='Train Loss')
 plt.plot(all_val_dice_loss, label='Validation Loss')
 plt.xlabel('Epochs')
 plt.ylabel('Loss')
 plt.legend()
 plt.title('Training and Validation Dice Loss')
 plot_path = os.path.join(out_dir, 'loss_plot.png')
 plt.savefig(plot_path)
 plt.close()
 print(f"Plot saved at: {plot_path}")

