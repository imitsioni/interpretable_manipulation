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
'num_workers' : 2,
'imgs_out' : 3,
'clstm_layers' : 3,
'clstm_hidden_units' : 32,
'clstm_stride' : 1,
'clstm_kernel_size' : (3,1),
'model_name' : None
}

#overwrite any defaults with given arguments
config, args = utils.load_args(config)

for config["imgs_out"] in [5]:#[3,5,7]:
    for config["clstm_layers"] in [3]:#[1,3]:
        for config["clstm_hidden_units"] in [32]:#[16,32,64]:
            for config["clstm_kernel_size"] in [(1,5)]:#[(1,3),(1,5)]:
        
                config['model_name'] = "CLSTM_"+"_im"+str(config["imgs_out"])+"_cl"+str(config['clstm_layers']) \
                                       + "_chu"+str(config['clstm_hidden_units'])+"_cks"+str(config['clstm_kernel_size'])

                if(config['model_name']==None):
                    print("Error, model name to save the checkpoint for must be given")
                    sys.exit(-1)

                # initialize dataloaders
                train_dl = dui.EventDataloader(config['train_path'], dataType="train", block_size = config['block_size'],\
                                    scale_pixels=config['scale_pixels'], block_horizon=config['horizon'], colormap=config['colormap'],imgs_out=config['imgs_out'])

                train_loader = torch.utils.data.DataLoader(train_dl, batch_size=config['batch_size'], shuffle=False,\
                                                           num_workers=config['num_workers'], pin_memory=True, drop_last=True)

                # Pass the training data scaler params to the testing loader
                scaler = train_dl.dh.scaler
                gradcam_scaler_params = train_dl.dh.gradcam_scaler_params


                test_dl = dui.EventDataloader(config['test_path'], dataType="test", block_size = config['block_size'],\
                                              scaler = scaler, gradcam_scaler_params = gradcam_scaler_params,\
                                              scale_pixels=config['scale_pixels'], block_horizon=config['horizon'],\
                                              colormap=config['colormap'],imgs_out=config['imgs_out'])

                test_loader = torch.utils.data.DataLoader(test_dl, batch_size=config['batch_size'], shuffle=False,\
                                                          num_workers=config['num_workers'], pin_memory=True, drop_last=True)

                criterion = nn.CrossEntropyLoss()

                if(torch.cuda.is_available()):
                    device = torch.device("cuda") 
                else:
                    device = torch.device("cpu")

                #model definition used in the work
                model = networks.CLSTMEventNet(num_classes = 2, image_size=(9,10),\
                           channels=3,conv_kernel_size=config["clstm_kernel_size"],\
                           top_layer=True, avg_pool=False, batch_normalization = False,\
                           lstm_layers=config["clstm_layers"],step=config['imgs_out'],\
                           conv_stride=1,device=device,\
                           max_pool=False,
                           effective_step=range(config['imgs_out']),\
                           nb_lstm_units=config["clstm_hidden_units"]).to(device).float()

                optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

                #Instantiate the trainer 
                trainer = train_utils.TrainUtils(model, train_loader, test_loader, optimizer, criterion, device)

                #optionally resume training
                if(args.resume):
                    trainer.loadModel(args.checkpoint)
                    model=trainer.model

                t_start = time.time()
                train_losses = []
                validation_losses = []
                train_F1s = []
                validation_F1s = []
                for epoch in range(config['epochs']):  
                    train_loss,trainF1 = trainer.train(epoch)
                    validation_loss, validationF1 = trainer.evaluate(model_title=config['model_name'])
                    train_losses.append(train_loss)
                    train_F1s.append(trainF1)
                    validation_losses.append(validation_loss) 
                    validation_F1s.append(validationF1)

                t_end = time.time()
                print('Finished Training , took ', (t_end - t_start))

                trainer.saveModel(config['model_name'],train_losses,validation_losses,train_F1s,validation_F1s,config)