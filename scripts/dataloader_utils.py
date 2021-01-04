import torch
from dataset_handlers import datasethandler
import torch.utils.data as udata
import numpy as np

class EventDataloader(udata.Dataset):
    '''
    Underlying dataset handler for the data.
    Arguments:
        base_path: Path to data
        dataType: The type of data to be loaded, one of "train", "test", "visual"
        block_size: Length M of the sequence comprising each block. 10 in implementation.
        scale_pixels: If the feature input representation should be scaled up (integer, factor of upscale, 1 in implementation)
        blocks_per_frame: How many blocks each input frame should include. 1 in implementation.
        multi_class: If the task should be multi-class classification. False in implementation.
        block_horizon: How many blocks ahead the "stuck/ not stuck" label is referring to. 3 in implementation.
        mono_color: If True, does not render a colored image for the data, only a 1 channel matrix. False in implementation.
        colormap: Which colormap to use for the RGB representation if mono_color is False. Seismic in implementation.
        scaler: scaler used for input data, if already computed during training.
        gradcam_scaler_params: scaler used for visually interpretable data display, if already computed during training.
        imgs_out: used to define how many images one input corresponds to. Default is 1, can be extended for recurrent models.
    '''
    def __init__(self, base_path, dataType, block_size, scale_pixels = 1, blocks_per_frame = 1, multi_class = False, block_horizon = 3, colormap="seismic", mono_color=False,scaler=None, gradcam_scaler_params=None,imgs_out=1):

        self.base_path = base_path
        self.blocks_per_frame = blocks_per_frame
        self.block_size = block_size
        self.scale_pixels = scale_pixels
        self.multi_class = multi_class
        self.block_horizon = block_horizon
        self.colormap=colormap
        self.dataType=dataType
        self.scaler = scaler
        self.imgs_out = imgs_out
        self.gradcam_scaler_params = gradcam_scaler_params
        if(self.imgs_out>1):
            self.blocks_per_frame = self.imgs_out #get more blocks, split to frames later
            
        self.dh = datasethandler.TXTImporter(base_path = self.base_path, dataType=self.dataType, block_size = self.block_size, scale_pixels = self.scale_pixels, blocks_per_frame = self.blocks_per_frame, multi_class = self.multi_class, block_horizon = self.block_horizon, colormap=self.colormap, scaler = self.scaler, gradcam_scaler_params = self.gradcam_scaler_params,mono_color=mono_color)

    def __len__(self):
        if(self.imgs_out>1):
            return len(self.dh.idxToRunDict)- self.imgs_out
        else:
            return len(self.dh.idxToRunDict)

    def __getitem__(self, idx):
        
        #zip together a sequence of input images and their true labels if requested (used for recurrent models)
        if(self.imgs_out>1):
            
            outs = []
            if(self.dataType=="visual"):
                for idxi in range(idx,idx+self.imgs_out):
                    out_net = self.dh.getFrame(idxi)
                    out_vis = self.dh.getVisIntFrame(idxi)
                    out = [out_net, out_vis]
                    outs.append(out)
                    
                outF_and_labels, out_viz_frames = zip(*outs)
                out_net_frames,labels = zip(*outF_and_labels)

                out_net_frames = np.array(out_net_frames)
                out_viz_frames = np.array(out_viz_frames)

                #reshape to have channel first, then sequence length
                out_net_frames = out_net_frames.reshape(3,self.imgs_out,9,self.block_size)
                out_viz_frames = out_viz_frames.reshape(3,self.imgs_out,9,self.block_size)
                
                return [(out_net_frames,labels[-1]),out_viz_frames]
            
            else:
                for idxi in range(idx,idx+self.imgs_out):
                    out = self.dh.getFrame(idx)
                    outs.append(out)
            
                out_frames, labels = zip(*outs)
                out_frames = np.array(out_frames)
                #reshape to have channel first, then sequence length
                out_frames = out_frames.reshape(3,self.imgs_out,9,self.block_size)
                
                #ground truth label for a sequence is the label of the last image
                return [out_frames,labels[-1]]
            
        #otherwise just return one image as the datapoint
        else:
            if(self.dataType=="visual"):
                out_net = self.dh.getFrame(idx)
                out_vis = self.dh.getVisIntFrame(idx)
                out = [out_net, out_vis]
            else:
                out = self.dh.getFrame(idx)
        return out
