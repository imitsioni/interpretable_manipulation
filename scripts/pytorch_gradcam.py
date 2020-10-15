'''
Code adapted from https://github.com/jacobgil/pytorch-grad-cam/blob/master/gradcam.py
'''

import torch
from torch.autograd import Variable
from torch.autograd import Function
import cv2
import sys
import numpy as np
import argparse
import math

class FeatureExtractor():
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """
    def __init__(self, model, target_layers,archType):
        self.model = model
        self.archType=archType
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():

            if (self.archType == "normal" and name == "fc1"):
                x = module(x.view(-1,self.model.channels[-1]*self.model.h3*self.model.w3))
            elif (self.archType == "small" and name == "fc1"):
                x = module(x.view(-1,self.model.channels[-1]*self.model.h1*self.model.w1))
            elif (self.archType == "old" and name == "fc1"):
                x = module(x.view(-1,128*9*10))
            else:
                x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]

        return outputs, x

class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """
    def __init__(self, model, target_layers,archType):
        self.model = model
        self.feature_extractor = FeatureExtractor(self.model, target_layers,archType)


    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations, output  = self.feature_extractor(x)
        output = output.view(output.size(0), -1)

        return target_activations, output

def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
    heatmap = heatmap[:,:,::-1]
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    temp = np.uint8(255 * cam)
    return temp

class GradCam:
    def __init__(self, model, target_layer_names, use_cuda,archType="normal"):
        self.model = model
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, target_layer_names,archType)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index = None):
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)
        one_hot[0][index] = 1
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad = True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()
        target = features[-1]
        target = target.cpu().data.numpy()[0, :]
        weights = np.mean(grads_val, axis = (2, 3))[0, :]
        cam = np.zeros(target.shape[1 : ], dtype = np.float32)

        for i, w in enumerate(weights):
            cam += np.maximum(w * target[i, :, :],0)

        cam = cv2.resize(cam, (input.shape[3], input.shape[2])) 

        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam


class FeatureEvaluator(object):
    '''
    Class used for evaluating the importance of features by using the GradCam images. Can return the number of times a certain feature was above a threshold, as well as the number of times a feature was most important.
    '''
    def __init__(self,featureThreshold=0.5, numFeatures = 9):
        '''
        Inputs:
            featureThreshold: The threshold a feature must be above (0-1) to be included in the "above threshold" count
            numFeatures: How many features the input image / gradcam image contains
        '''
        self.featureCounts = {'TP':[0,0,0,0,0,0,0,0,0], 'TN':[0,0,0,0,0,0,0,0,0], 'FP':[0,0,0,0,0,0,0,0,0], 'FN':[0,0,0,0,0,0,0,0,0]}
        self.featureCountsMax = {'TP':[0,0,0,0,0,0,0,0,0], 'TN':[0,0,0,0,0,0,0,0,0], 'FP':[0,0,0,0,0,0,0,0,0], 'FN':[0,0,0,0,0,0,0,0,0]}
        self.caseCounts = {'TP' : 0, 'TN' : 0, 'FP' : 0, 'FN' : 0}
        self.featureThreshold = featureThreshold
        self.numFeatures = numFeatures

    def reset(self):
        self.featureCounts = {'TP':[0,0,0,0,0,0,0,0,0], 'TN':[0,0,0,0,0,0,0,0,0], 'FP':[0,0,0,0,0,0,0,0,0], 'FN':[0,0,0,0,0,0,0,0,0]}
        self.featureCountsMax = {'TP':[0,0,0,0,0,0,0,0,0], 'TN':[0,0,0,0,0,0,0,0,0], 'FP':[0,0,0,0,0,0,0,0,0], 'FN':[0,0,0,0,0,0,0,0,0]}
        self.caseCounts = {'TP' : 0, 'TN' : 0, 'FP' : 0, 'FN' : 0}

    def update(self,predClass,GT,gradCamImage):
        '''
        Updates the current results with a new sample.
        Inputs:
            PredClass: The predicted class for the sample
            GT: The ground truth for the sample
            gradCamImage: The produced GradCam image for the sample
        '''
        if(predClass==0 and GT == 0):
            currentCase = 'TN'
        elif(predClass==0 and GT == 1):
            currentCase = 'FN'
        elif(predClass==1 and GT == 0):
            currentCase = 'FP'
        elif(predClass==1 and GT == 1):
            currentCase = 'TP'

        self.caseCounts[currentCase] += 1

        #In case the image has been scaled up, calculate row offset in pixels
        rowFactor = gradCamImage.shape[0]/self.numFeatures

        for f_idx in range(self.numFeatures):
            if(any(gradCamImage[int(rowFactor*f_idx)]>self.featureThreshold)):
                self.featureCounts[currentCase][f_idx] += 1

        maxValFeature = np.unravel_index(np.argmax(gradCamImage, axis=None), gradCamImage.shape)[0]
        self.featureCountsMax[currentCase][maxValFeature] += 1
        
    def returnResults(self):
        return self.featureCountsMax, self.featureCounts, self.caseCounts

    def showResults(self):
        for case in self.featureCounts.keys():
            print("out of %d %s cases, features above %f were: "%(self.caseCounts[case], case, self.featureThreshold))
            print("P_x: " , self.featureCounts[case][0])
            print("P_y: " , self.featureCounts[case][1])
            print("P_z: " , self.featureCounts[case][2])
            print("F_x: " , self.featureCounts[case][3])
            print("F_y: " , self.featureCounts[case][4])
            print("F_z: " , self.featureCounts[case][5])
            print("U_x: " , self.featureCounts[case][6])
            print("U_y: " , self.featureCounts[case][7])
            print("U_z: " , self.featureCounts[case][8])
            print("-----------------")

        print("____________________________________________")
        for case in self.featureCounts.keys():
            print("out of %d %s cases, the most cared about feature was: "%(self.caseCounts[case], case))
            print("P_x: " , self.featureCountsMax[case][0])
            print("P_y: " , self.featureCountsMax[case][1])
            print("P_z: " , self.featureCountsMax[case][2])
            print("F_x: " , self.featureCountsMax[case][3])
            print("F_y: " , self.featureCountsMax[case][4])
            print("F_z: " , self.featureCountsMax[case][5])
            print("U_x: " , self.featureCountsMax[case][6])
            print("U_y: " , self.featureCountsMax[case][7])
            print("U_z: " , self.featureCountsMax[case][8])
            print("-----------------")

