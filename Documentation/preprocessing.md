Preprocessing Steps 

1) extract data from hdf5 file using dat2d_ucsf_1d.py script 

2) convert to numpy array 

3) if reshape desired, use the resize_images function from the image_ops.py script 

4) normalize both train and test images 

5) if positive bool is set to true, all corrosponding images and masks that do not contain tumors are filtered out 

6) if augmentation bool is set to true, the specified percent of images are randomely augmented 

7) add a singleton dimension in masks for channels (always 1 in masks)

8) create train and test loaders using the data.Dataloader class 

9) convert into torch tensors and run trainig, testing or inference 