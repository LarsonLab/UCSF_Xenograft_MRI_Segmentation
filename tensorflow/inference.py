import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from plot import save_plots
from metrics import dice_coef
from r2udensenet_1d import r2udensenet
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from data2D_ucsf_1d import load_test_data


inference_dir = './logs/j2'
if not os.path.exists(inference_dir):
    os.makedirs(inference_dir)


# mask_train = mask_train.astype('float64')

model = r2udensenet()



# mask_train /= 255.
model.load_weights("/data/ernesto/UCSF-Prostate-Segmentation/logs/Training_10_11_2023/plots1/weights/saved-model-186-0.39.hdf5")

images_test = load_test_data()
images_test = images_test.astype('float64')
images_test_mean = np.mean(images_test)
images_test_std = np.std(images_test)
images_test = (images_test - images_test_mean)/images_test_std
images_test = images_test.reshape(images_test.shape[0],192,192,1)



predict = model.predict(images_test)

#add dice coeff and iou 
#add ground truth

# predict_images = predict.shape[0]
# images_train = images_train

for i in range(len(images_test)):
    orginal_image = images_test[i].reshape(192,192)
    predict_image = predict[i].reshape(192,192)
    


    plt.figure()
    plt.imshow(orginal_image,cmap = 'gray')
    plt.imshow(predict_image,cmap = 'viridis',alpha=0.4)
    plt.title('test')
    plt.savefig(os.path.join(inference_dir + "/test" +str(i+1)+".png"))
    # trainMask = mask_train[image]