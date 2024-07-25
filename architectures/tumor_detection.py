import torch 
import torch.nn as nn 
import torchvision 
import numpy as np 
import os 
from skimage.io import imsave 
import random 
from torchvision.models import resnet18
from torch.optim import SGD
from copy import deepcopy
from tqdm import tqdm
import datetime


seed = 42 
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
weights_dir_path = '/home/henry/UCSF_Prostate_Segmentation/Weights/Resnet18_weights/'
current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
weights_save_path = os.path.join(weights_dir_path,f'{current_time}.pth')





def train_resnet_classification(device,train_loader,validation_loader,test_loader,epochs,train_set_length,val_set_length,batch_size):

    net = resnet18()
    net.conv1 = nn.Conv2d(1,64,kernel_size=(7,7),stride=(2,2),padding=(3,3),bias=False)
    net.fc = nn.Linear(net.fc.in_features,2)
    net = net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(net.parameters(),lr=0.001)
    epochs = 100 
    net_final = deepcopy(net)

    best_validation_accuracy = 0 
    train_accs = []
    val_accs = []
    print('starting detection training')

    for epoch in range(epochs):
        net.train()
        print('### Epoch {}'.format(epoch))
        total_train_examples = 0 
        num_correct_train = 0 
        for i,batch in tqdm(enumerate(train_loader),total=train_set_length//batch_size):
            inputs = batch[0].float().to(device)
            targets = batch[1].long().to(device)
            optimizer.zero_grad()
            predictions = net(inputs)
            print('yes' if torch.argmax(predictions[0]) == 1 else 'no')
            loss = criterion(predictions,targets)
            loss.backward()
            optimizer.step()
            _, predicted_class = predictions.max(1)
            total_train_examples += predicted_class.size(0)
            num_correct_train += predicted_class.eq(targets).sum().item()

        train_acc = num_correct_train / total_train_examples 
        print('Training accuracy: {}'.format(train_acc))
        train_accs.append(train_acc)

        total_val_examples = 0 
        num_correct_val = 0 

        net.eval()

        with torch.no_grad():
            for batch_index,(inputs,targets) in tqdm(enumerate(validation_loader),total=val_set_length//batch_size):
                inputs = inputs.float().to(device)
                targets = targets.float().to(device)
                predictions = net(inputs)
                _, predicted_class = predictions.max(1)
                total_val_examples += predicted_class.size(0)
                num_correct_val += predicted_class.eq(targets).sum().item()

        val_acc = num_correct_val / total_val_examples 
        print('Validation accuracy: {}'.format(val_acc))
        val_accs.append(val_acc)

        if val_acc > best_validation_accuracy: 
            best_validation_accuracy = val_acc 
            print('Validation accuracy improved; saving model.')
            net_dictionary = net.state_dict()

    torch.save(net_dictionary,weights_save_path)


        
def detect_tumors(device,model,dataset_loader,weights_path,dataset_length,deploy: bool):

    if deploy:

        detections = []
        positive_samples = []
        negative_samples = []
        all_images = list(dataset_loader)
        model = model
        model.load_state_dict(torch.load(weights_path))
        model = model.to(device)
        model.eval()
        
        with torch.no_grad():
            for batch_index,(inputs,discard) in tqdm(enumerate(dataset_loader),total=dataset_length,desc='Detecting Tumors'):
                inputs = inputs.to(device)
                predictions = model(inputs)
                _,predicted_class = predictions.max(1)
                detections.append(predicted_class.item())

        for i in range(len(detections)-1):
            if detections[i] == 0: 
                negative_samples.append(all_images[i])
            else:
                positive_samples.append(all_images[i])

        positive_samples = np.array(positive_samples)
        negative_samples = np.array(negative_samples)

        print(f'Positive Samples: {len(positive_samples)}\nNegative Samples: {len(negative_samples)}')

        return positive_samples,negative_samples

    else: 

        with torch.no_grad():

            total_test_examples = 0
            true_pos_count = 0 
            false_pos_count = 0 
            model = resnet18()
            model = model.load_state_dict(torch.load(weights_path))
            model = model.to(device)
            model.eval()

            for batch_index,(inputs,targets) in tqdm(dataset_loader,total=dataset_length,desc='Testing Detection Model'):
                inputs = inputs.to(device)
                targets = targets.to(device)
                predictions = model(inputs)
                _,predicted_class = predictions.max(1)
                total_test_examples += predicted_class.size(0)
                num_correct_test += predicted_class.eq(targets).sum().item()
                confusion_vector = predicted_class / targets
                num_true_pos = torch.sum(confusion_vector == 1).item()
                num_false_pos = torch.sum(confusion_vector == float('inf')).item()

                true_pos_count += num_true_pos
                false_pos_count += num_false_pos 

        test_acc = num_correct_test / total_test_examples
        print(f'True set accuracy: {test_acc}')
        print(f'True positive classifications:{true_pos_count}\nFalse positive classifications:{false_pos_count}')







        



        











