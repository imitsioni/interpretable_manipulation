'''
MISC Utilities.
'''
import matplotlib.pyplot as plt
import numpy as np
import os
import fnmatch
import sys
import warnings
import argparse

def reset_dataset(position_data, force_data, f_ref_data):
    combined_dict = {}
    for dataset_id in position_data.keys():
        combined_dict.update({dataset_id : np.concatenate((position_data[dataset_id], force_data[dataset_id], f_ref_data[dataset_id]), axis = 1)})
    return combined_dict

# Breaking the datasets in non-overlaping blocks
def get_block(data, idx, block_size):
    from_idx = idx * block_size;
    to_idx = (idx + 1) *  block_size -1;
    tot_blocks = int(np.floor(data.shape[0]/block_size))
    success = 1
    # check if idx is out of bounds
    if (to_idx > data.shape[0]):
        warnings.warn("Not enough datapoints remaining to make a block.")
        success = 0
    if success:
        block = data[from_idx:to_idx + 1, :]
    else:
        block = 666*np.ones((block_size, data.shape[1]))
    return block

def get_relative_position(current_block, previous_block):
    relative = np.copy(current_block)
    relative[:,:3] = relative[:,:3] - previous_block[-1,:3]
    return relative

def get_relative_dataset(dataset, block_size):
    previous_block =  get_block(dataset, 0, block_size)
    relative_dataset = np.empty((1,dataset.shape[1])) # it will have an extra row, remove it
    tot_blocks = int(np.floor(dataset.shape[0]/block_size))

    for i in range(1, tot_blocks):
        current_block = get_block(dataset,i, block_size)
        relative_dataset = np.append(relative_dataset, get_relative_position(current_block, previous_block), axis = 0)
        previous_block = get_block(dataset, i, block_size)

    relative_dataset = relative_dataset[1:,]
    return relative_dataset

def get_feature_abs_max(data):
    data_copy = np.copy(data)
    colmax = np.max(data_copy, axis = 0)
    colmin = np.min(data_copy, axis = 0)
    abs_max = np.max(colmax)
    abs_min = np.min(colmin)
    return [abs_min, abs_max]

def load_args(defaultConfig):
    
    #load args
    parser = argparse.ArgumentParser(description='Training script for EventNet')

    parser.add_argument('--resume', '-r', action='store_true',
                        help="resume training from a given checkpoint.")
    parser.add_argument('--checkpoint', '-chp', type=str,
                        help="model name to resume training from")
    parser.add_argument('--learning_rate', '-lr', type=float,
                        help="initial learning rate")
    parser.add_argument('--batch_size', '-bs', type=int,
                        help="batch size for training")
    parser.add_argument('--num_workers', '-nw', type=int,
                        help="worker threads for dataloading")
    parser.add_argument('--epochs', '-ep', type=int,
                        help="how many epochs to train")
    parser.add_argument('--trainPath', '-trnp', type=str,
                        help="Path to training data")
    parser.add_argument('--testPath', '-tstp', type=str,
                        help="Path to training data")    
    parser.add_argument('--horizon', '-hz', type=int,
                        help="How many blocks ahead class label refers to")
    parser.add_argument('--block_size', '-blks', type=int,
                        help="Length M of the sequence comprising each block.")
    parser.add_argument('--scale_pixels', '-pscl', type=int,
                        help="upscaling factor for input representation.")
    parser.add_argument('--colormap', '-cmap', type=str,
                        help="colormap to be used for input rendering")
    parser.add_argument('--modelName', '-m', type=str,
                        help="What the saved model should be called")
        
    args = parser.parse_args()
    
    if len(sys.argv) < 2: #model name must at least be given
        parser.print_help()
        sys.exit(1)
        
    #overwrite defaults with any given arguments
    if(args.learning_rate is not None):
        defaultConfig['learning_rate'] = args.learning_rate
    if(args.batch_size is not None):
        defaultConfig['batch_size'] = args.batch_size
    if(args.num_workers is not None):
        defaultConfig['num_workers'] = args.num_workers
    if(args.epochs is not None):
        defaultConfig['epochs'] = args.epochs
    if(args.trainPath is not None):
        defaultConfig['train_path'] = args.trainPath
    if(args.testPath is not None):
        defaultConfig['test_path'] = args.testPath
    if(args.horizon is not None):
        defaultConfig['horizon'] = args.horizon
    if(args.block_size is not None):
        defaultConfig['block_size'] = args.block_size
    if(args.scale_pixels is not None):
        defaultConfig['scale_pixels'] = args.scale_pixels
    if(args.colormap is not None):
        defaultConfig['colormap'] = args.colormap
    if(args.modelName is not None):
        defaultConfig['model_name'] = args.modelName

    return defaultConfig,args