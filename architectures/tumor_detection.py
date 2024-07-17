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


seed = 42 
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)





def build_resnet(device,train_loader,validation_loader,test_loader,epochs,train_set_length,val_set_length,batch_size):

    net = resnet18()
    net.conv1 = nn.Conv2d(1,64,kernel_size=(7,7),stride=(2,2),padding=(3,3),bias=False)
    net = net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(net.parameters(),lr=0.001)
    epochs = 100 
    net_final = deepcopy(net)

    best_validation_accuracy = 0 
    train_accs = []
    val_accs = []

    for epoch in range(epochs):
        net.train()
        print('### Epoch {}'.format(epoch))
        total_train_examples = 0 
        num_correct_train = 0 
        for batch_index,(inputs,targets) in tqdm(enumerate(train_loader),total=train_set_length//batch_size):
            inputs = inputs.float().to(device)
            targets = targets.float().to(device)
            optimizer.zero_grad()
            predictions = net(inputs)
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
            for batch_index,(inputs,targets) in tqdm(enumerate(train_loader),total=val_set_length//batch_size):
                inputs = inputs.float().to(device)
                targets = targets.float().to(device)
                predictions = net(inputs)
                _, predicted_class = predictions.max(1)
                total_val_examples += predicted_class.size(0)
                num_correct_val += predicted_class.eq(targets).sum().item()

        val_acc = num_correct_val / total_val_examples 
        print('Validation accuracy: {}'.format(val_acc))
        val_accs.append(val_acc)


        











