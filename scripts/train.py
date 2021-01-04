#Imports
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time 
import importlib

import utils
import image_render_code as imgutils
import dataloader_utils as dui
import pytorch_gradcam
import train_utils
import networks

from matplotlib.colors import LinearSegmentedColormap

#Default Paths and Experiment Configuration
#These values are from the original experiment for the 2D version.
config = {
'train_path' : '../data/cutting_datasets/train',
'test_path' : '../data/cutting_datasets/test',
'horizon' : 3, #How many blocks ahead class label refers to
'block_size' : 10, #Length M of the sequence comprising each block.
'scale_pixels' : 1, #upscaling factor for input representation (integer)
'colormap' : "seismic", #colormap to be used for input rendering
'epochs' : 25,
'learning_rate' : 1e-03 ,
'batch_size' : 32,
'num_workers' : 8,
'model_type' : "cnn", #Which model to create (cnn,FC,LSTM)
'model_name' : None
}

#overwrite any defaults with given arguments
config, args = utils.load_args(config)
if(config['model_name']==None):
    print("Error, model name to save the checkpoint for must be given")
    sys.exit(-1)

# initialize dataloaders
if(config['model_type']!="cnn"):
    mono_color=True
else:
    mono_color=False
    
train_dl = dui.EventDataloader(config['train_path'], dataType="train", block_size = config['block_size'],\
                    scale_pixels=config['scale_pixels'], block_horizon=config['horizon'],\
                               colormap=config['colormap'],mono_color=mono_color)

train_loader = torch.utils.data.DataLoader(train_dl, batch_size=config['batch_size'], shuffle=True,\
                                           num_workers=config['num_workers'], pin_memory=True, drop_last=True)

# Pass the training data scaler params to the testing loader
scaler = train_dl.dh.scaler
gradcam_scaler_params = train_dl.dh.gradcam_scaler_params


test_dl = dui.EventDataloader(config['test_path'], dataType="test", block_size = config['block_size'],\
                              scaler = scaler, gradcam_scaler_params = gradcam_scaler_params,\
                              scale_pixels=config['scale_pixels'], block_horizon=config['horizon'],\
                              colormap=config['colormap'],mono_color=mono_color)

test_loader = torch.utils.data.DataLoader(test_dl, batch_size=config['batch_size'], shuffle=True,\
                                          num_workers=config['num_workers'], pin_memory=True, drop_last=True)

criterion = nn.CrossEntropyLoss()

if(torch.cuda.is_available()):
    device = torch.device("cuda") 
else:
    device = torch.device("cpu")
print(config["model_type"])

#model definitions used in the work
if(config["model_type"]=="FC"):
    #Create the default Fully Connected Network
    model = networks.EventNetFC(inputImgSize=[9,10],LayerSizes=[90,90,90],classes=2, channels_in=1).to(device).float()

elif(config["model_type"]=="LSTM"):
    #Create the default LSTM Network
    model = networks.EventNetLSTM(inputImgSize=[9,10],LayerSizes=[9,9,9],classes=2, channels_in=1,layers=3).to(device).float()

else:
    #Convolutional model used for the 2D experiments.
    #Original net is f= (1,5), (1,3), (1,1), stride 1, pads 0 all around
    model = networks.MultiEventNetTemplate(classes = 2, inputImgSize=[9,10], channels_in=3,channels=[32,64,128],\
                                           filterSizes=[(5,5),(3,3),(2,2)],strides=[(1,1),(1,1),(1,1)],\
                                           paddings=[(5//2,0//2),(3//2,0//2),(2//2,0//2)],biases=True).to(device).float()

optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

#Instantiate the trainer 
trainer = train_utils.TrainUtils(model, train_loader, test_loader, optimizer, criterion, device)

#optionally resume training
if(args.resume):
    trainer.loadModel(args.checkpoint)
    model=trainer.model

train_losses = []
validation_losses = []
train_F1s = []
validation_F1s = []

for epoch in range(config['epochs']):  
    train_loss,trainF1 = trainer.train(epoch)
    t_start = time.time()
    validation_loss, validationF1 = trainer.evaluate(model_title=config['model_name'])
    train_losses.append(train_loss)
    train_F1s.append(trainF1)
    validation_losses.append(validation_loss) 
    validation_F1s.append(validationF1)      

t_end = time.time()
print('Finished Training , took ', (t_end - t_start))

#save resulting model along with the config used for the experiment, and training/evaluation scores
trainer.saveModel(config['model_name'],train_losses,validation_losses,train_F1s,validation_F1s,config)