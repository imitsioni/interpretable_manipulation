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
        colormap: Which colormap to use for the RGB representation. Seismic in implementation.
        scaler: scaler used for input data, if already computed during training
        gradcam_scaler_params: scaler used for visually interpretable data display, if already computed during training
    '''
    def __init__(self, base_path, dataType, block_size, scale_pixels = 1, blocks_per_frame = 1, multi_class = False, block_horizon = 3, colormap="seismic", scaler=None, gradcam_scaler_params=None):

        self.base_path = base_path
        self.blocks_per_frame = blocks_per_frame
        self.block_size = block_size
        self.scale_pixels = scale_pixels
        self.multi_class = multi_class
        self.block_horizon = block_horizon
        self.colormap=colormap
        self.dataType=dataType
        self.scaler = scaler
        self.gradcam_scaler_params = gradcam_scaler_params

        self.dh = datasethandler.TXTImporter(base_path = self.base_path, dataType=self.dataType, block_size = self.block_size, scale_pixels = self.scale_pixels, blocks_per_frame = self.blocks_per_frame, multi_class = self.multi_class, block_horizon = self.block_horizon, colormap=self.colormap, scaler = self.scaler, gradcam_scaler_params = self.gradcam_scaler_params)

    def __len__(self):
        return len(self.dh.idxToRunDict)

    def __getitem__(self, idx):
        if(self.dataType=="visual"):
            out_net = self.dh.getFrame(idx)
            out_vis = self.dh.getVisIntFrame(idx)
            out = [out_net, out_vis]
        else:
            out = self.dh.getFrame(idx)
        return out
