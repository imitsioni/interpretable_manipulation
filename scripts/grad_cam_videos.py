'''
Code adapted from https://github.com/TwentyBN/smth-smth-v2-baseline-with-models/blob/master/grad_cam_videos.py
'''
import sys
import torch
import cv2
import numpy as np

from torch.autograd import Variable


from pytorch_gradcam import *


class ModelOutputsVideo(ModelOutputs):
    """ 
    Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermediate targetted layers.
    3. Gradients from intermediate targetted layers. 
    """

    def __init__(self, model, target_layers, archType):
        self.model = model
        self.archType = archType
        self.feature_extractor = FeatureExtractor(
            self.model, target_layers, archType)
        
        self.smlayer = torch.nn.Softmax()

    def __call__(self, x):
        target_activations, output = self.feature_extractor(x)
            
        return target_activations, output


class GradCamVideo(GradCam):
    def __init__(self, model, target_layer_names, class_dict, use_cuda,
                 input_spatial_size=224, normalizePerFrame=False, archType="CLSTM"):
        self.model = model

        self.archType=archType
        self.cuda = use_cuda
        self.normalizePerFrame = normalizePerFrame
        if(len(input_spatial_size)==1):
            self.input_spatial_size = (input_spatial_size,input_spatial_size)
        else:
            self.input_spatial_size = input_spatial_size
        self.class_dict = class_dict
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputsVideo(self.model, target_layer_names,self.archType)

    def __call__(self, input, index=None):
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)

        if index is None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad=True)
        
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.model.zero_grad()

        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()
        
        # Need to put sequence dimension in correct spot again if using pytorch CLSTM
        if(self.archType=="CLSTM"):
            grads_val = grads_val.transpose(1,2,0,3,4)
            target = features[-1].permute(1,2,0,3,4)
        else:
            target = features[-1]
            
        target = target.cpu().data.numpy()[0, :]
        weights = np.mean(grads_val, axis=(2, 3, 4))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += np.maximum(w * target[i, :, :, :],0)

        clip_size = input.size(2)
        step_size = clip_size // target.shape[1]

        cam_vid = []
        for i in range(cam.shape[0]):
            cam_map = cam[i, :, :]
            cam_map = cv2.resize(cam_map,
                                 (self.input_spatial_size[1], self.input_spatial_size[0]))#WARNING: was 0,1
            cam_vid.append(np.repeat(
                                np.expand_dims(cam_map, 0),
                                step_size,
                                axis=0)
                           )

        cam_vid = np.array(cam_vid)
        
        if(self.normalizePerFrame):
            for i in range(cam_vid.shape[0]):
                cam_vid[i] = cam_vid[i] - np.min(cam_vid[i])
                cam_vid[i] = cam_vid[i] / np.max(cam_vid[i])
        else:
            cam_vid = cam_vid - np.min(cam_vid)
            cam_vid = cam_vid / np.max(cam_vid)
            
        if cam_vid.shape[0] > 1:
            cam_vid = np.concatenate(cam_vid, axis=0)
        if cam_vid.shape[0] == 1:
            cam_vid = np.squeeze(cam_vid, 0)

        return cam_vid, output