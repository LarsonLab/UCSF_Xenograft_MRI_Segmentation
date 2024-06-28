import torch 
import torchvision 
import torch.nn as nn 
import numpy as np 
#import pandas as pd
from torchvision import utils as  vutils 
from torchvision import transforms 
from torch import nn 
from torch.nn import functional as F 
from torch.utils import data 
from torch.optim import SGD, Adam 
from PIL import Image 
#import matplotlib.pyplot as plt 
import os 
from statistics import mean 
from torchsummary import summary 
#from sklearn.metrics import roc_auc_score  as AUC
#from sklearn.metrics import average_precision_score, precision_recall_curve 


class Convolution(nn.Module): 

    def __init__(self,in_channels,out_channels): 
        super(Convolution,self).__init__()
        self.reconv1 = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,
                      padding=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.2,inplace=False)
        )

    def forward(self,x): 

        x = self.reconv1(x) #(1,32)

        return x
    

class Initial_RecLayer(nn.Module): 

    def __init__(self,in_channels,out_channels): 
        super(Initial_RecLayer,self).__init__()
        self.conv = Convolution(in_channels,out_channels)
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1,padding=0,bias=False)
        self.conv2 = Convolution(out_channels,out_channels)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.drop = nn.Dropout2d(p=0.2,inplace=False)


    def forward(self,x): 
        reconv1 = self.conv(x) #(1,32)
        layer_add = self.batch_norm(self.conv1(x)) #(1,32)
        add_conv1 = self.relu(reconv1 + layer_add) #(32,32)
        reconv2 = self.conv2(add_conv1) #(32,32)
        add_conv2 = self.relu(reconv2 + layer_add) #(32,32)
        reconv3 = self.conv2(add_conv2) #(32,32)
        reconv3 = self.drop(reconv3)

        return reconv3 
    

class RecLayer(nn.Module): 

    def __init__(self,in_channels,out_channels): 
        super(RecLayer,self).__init__()
        self.conv = Convolution(in_channels,in_channels)
        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size=1,padding=0,bias=False)
        self.batch_norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=False)
        self.drop = nn.Dropout(p=0.2,inplace=False)

    def forward(self,x): 
        reconv1 = self.conv(x) #(32,32)
        layer_add = self.batch_norm(self.conv1(x))
        add_conv1 = self.relu(reconv1 + layer_add )
        reconv2 = self.conv(add_conv1)
        add_conv2 = self.relu(reconv2 + layer_add )
        reconv3 = self.conv(add_conv2)
        reconv3 = self.drop(reconv3)


        return reconv3


image_row = 128 
image_col = 128
image_depth = 1

class DenseBlock(nn.Module): 

    def __init__(self,in_channels,out_channels): 
        super(DenseBlock,self).__init__()
        self.rec_layer = RecLayer(in_channels,in_channels)
        self.conv1x1 = nn.Conv2d(in_channels,in_channels,kernel_size=1,stride=1,
                                 padding=0,bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.batch_norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=False)

    def forward(self,x): 
        conv1 = self.rec_layer(x)
        conv1 = self.rec_layer(conv1)
        skip_connection = conv1
        conv1add = self.batch_norm(self.conv1x1(x))
        add1 = self.relu(conv1add + conv1)
        dense1 = torch.cat((add1,conv1),dim=1)
        pool1 = self.maxpool(dense1)
        return pool1,skip_connection
    
class Initial_DenseBlock(nn.Module): 

    def __init__(self,in_channels,out_channels): 
        super(Initial_DenseBlock,self).__init__()
        self.initial_rec = Initial_RecLayer(in_channels,out_channels)
        self.rec_layer = RecLayer(out_channels,out_channels)
        self.conv1x1 = nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1,
                                 padding=0,bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)

    def forward(self,x): 
        conv1 = self.initial_rec(x) #(1,32)
        conv1 = self.rec_layer(conv1) #(32,32)
        skip_connection = conv1
        conv1add = self.batch_norm(self.conv1x1(x))
        add1 = self.relu(conv1add + conv1)
        dense1 = torch.cat((add1,conv1),dim=1)
        pool1 = self.maxpool(dense1)

        return pool1,skip_connection

        
class DenseBlock_Drop(nn.Module): 

    def __init__(self,in_channels,out_channels): 
        super(DenseBlock_Drop,self).__init__()
        self.rec_layer = RecLayer(in_channels,in_channels)
        self.conv = nn.Conv2d(in_channels,in_channels,kernel_size=1,stride=1,
                              padding=0,bias=False)
        self.dropout = nn.Dropout2d(p=0.5,inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.batch_norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=False)

    def forward(self,x): 
        conv4 = self.rec_layer(x)
        conv4 = self.rec_layer(conv4)
        skip_connection = conv4 
        conv4add = self.batch_norm(self.conv(x))
        add4 = self.relu(conv4add + conv4)
        dense4 = torch.cat((add4,conv4),axis=1)
        drop4 = self.dropout(dense4)
        pool4 = self.maxpool(drop4)

        return pool4, skip_connection
    
    
class UpTransBlock(nn.Module):
    def __init__(self,in_channels,out_channels): 
        super(UpTransBlock,self).__init__()
        self.uptrans = nn.ConvTranspose2d(in_channels,out_channels,kernel_size=2,stride=2)

    def forward(self,x):
        return self.uptrans(x)
    
    
class BottleNeck(nn.Module): 

    def __init__(self,in_channels,out_channels): 
        super(BottleNeck,self).__init__()
        self.rec_layer = RecLayer(in_channels,in_channels)
        self.conv = nn.Conv2d(in_channels,in_channels,kernel_size=1,
                              stride = 1, padding=0,bias = False)
        self.dropout = nn.Dropout(p=0.5)
        self.batch_norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=False)

    def forward(self,x): 
        conv5 = self.rec_layer(x)
        conv5 = self.rec_layer(conv5)
        conv5add = self.batch_norm(self.conv(x))
        add5 = self.relu(conv5add + conv5)
        dense5 = torch.cat((add5,conv5),dim=1)
        drop5 = self.dropout(dense5)

        return drop5
    
class UpDenseBlock(nn.Module): 

    def __init__(self,in_channels,out_channels,cat_dim,skip_connection=None): 
        super(UpDenseBlock,self).__init__()
        self.up_trans = nn.ConvTranspose2d(in_channels,out_channels,kernel_size=2,stride=2,padding=0)
        self.rec_layer = Initial_RecLayer(cat_dim,out_channels)
        self.rec_layer1 = RecLayer(out_channels,out_channels)
        self.conv = nn.Conv2d(cat_dim,out_channels,kernel_size=1,
                              stride=1,padding=0,bias = False)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.skip_connection = skip_connection 
        
    def forward(self,x,skip_connection): 
        up1 = self.up_trans(x)
        cat1 = torch.cat((up1,skip_connection),dim=1)
        conv6 = self.rec_layer(cat1)
        conv6 = self.rec_layer1(conv6)
        conv6add = self.batch_norm(self.conv(cat1))
        add6 = conv6add + conv6
        dense6 = torch.cat((add6,conv6),dim=1)

        return dense6 
    
class OutputConv(nn.Module): 

    def __init__(self,in_channels,out_channels): 
        super(OutputConv,self).__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1,
                              padding=0,bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x): 
        conv10 = self.conv(x)
        out10 = self.sigmoid(conv10)

        return conv10
    

  

class r2udensenet(nn.Module): 

    def __init__(self,in_channels=1,out_channels=1): 
        super(r2udensenet,self).__init__()
        self.up_trans = nn.ConvTranspose2d(in_channels,out_channels,kernel_size=2,stride=2)
        self.encoder = nn.ModuleList([
            Initial_DenseBlock(in_channels,32),
            DenseBlock(64,128),
            DenseBlock(128,256),
            DenseBlock_Drop(256,512)
        ])
        self.bottleneck = BottleNeck(512,1024)
        self.decoder = nn.ModuleList([
            UpDenseBlock(1024,256,512),
            UpDenseBlock(512,128,256),
            UpDenseBlock(256,64,128),
            UpDenseBlock(128,32,64)
        ])
        self.output_conv = OutputConv(64,out_channels)

    def forward(self,x): 
        skips = []
        o = x
        for layer in self.encoder: 
            o,skip = layer(o)
            skips.append(skip) 
        torch.cuda.empty_cache()

        o = self.bottleneck(o)

        for i, layer in enumerate(self.decoder): 
            j = skips[len(skips)-i-1]
            o = layer(o,j)
        
        o = self.output_conv(o)
        torch.cuda.empty_cache()


        return(o)
        
pretrained_weights = None 

if(pretrained_weights): 
    try: 
        model = r2udensenet()
        model.load_state_dict(torch.load(pretrained_weights))
    except: 
        weights = torch.load(pretrained_weights)
        model_dict = model.state_dict()

        pretrained_dict = {k: v for k, v in pretrained_weights.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

 


    

        
            

        
        


    


        

    





#metrics section 

    



