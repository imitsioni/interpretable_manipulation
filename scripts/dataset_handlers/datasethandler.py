import os
import fnmatch
import sys
import utils
import numpy as np
import image_render_code as imgutils
import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn.preprocessing as scalers


class TXTImporter(object):
    '''
    Data handler for importing the training data from TXT.
    Inputs:
        base_path: Path to the training data
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
    def __init__(self, base_path, dataType, block_size, scale_pixels, blocks_per_frame, multi_class, block_horizon,colormap, scaler=None, gradcam_scaler_params=None):
        self.data = []
        self.base_path = base_path
        self.block_size = block_size
        self.colormap = colormap
        self.dataType = dataType
        # Initialization
        self.force_data = {}
        self.position_data = {}
        self.input_data = {}
        self.relative_datasets = {}
        self.gradcam_scaler_params = [] # e.g. gradcam_scaler_params[0][1] == max position
        self.idxToRunDict = {}

        self.blocks_per_frame = blocks_per_frame
        self.scale_pixels = scale_pixels
        self.cell_size = (self.scale_pixels, self.scale_pixels)

        self.label_threshold = 0.00005

        self.block_horizon = block_horizon
        self.multi_class = multi_class

        if(dataType!="train" and (scaler==None or gradcam_scaler_params==None)):
            print("Error, test and visual dataloaders must be provided with\
                    input data scalers and gradcam data scalers on initialization.")
            sys.exit(-1)

        # Read the data
        self.importDataset()

        if(self.dataType=="train"):
            #calculate scalers if running through training data
            self.scaler = scalers.MinMaxScaler()
            self.getGradCamScalerParams()
        else:
            #otherwise receive previously computed scalers
            self.gradcam_scaler_params = gradcam_scaler_params
            self.scaler = scaler
            relative_full = self.createRelativefull()

        self.createIdxToRunBlockDict()

    def importDataset(self, verbose = False):
        """
        The full dataset contains all the sub-datasets for each object and setting. Given the
        directory they are in, all txt files (for forces, positions, and reference forces) will
        be located and incorporated.
        """
        # Get a list of the various txt files
        listOfPositionFiles = [] # actual cartesian position
        listOfForceFiles = [] # sensor forces
        listOfInputFiles = [] # f_ref "forces"

        full_positions = np.empty((1,3))
        full_forces = np.empty((1,3))
        full_inputs = np.empty((1,3))

        for (dirpath, dirnames, filenames) in os.walk(self.base_path):
            dirnames.sort()

            for txtname in fnmatch.filter(filenames, 'positions.txt'):
                listOfPositionFiles.append(os.path.join(dirpath, txtname))

            for txtname in fnmatch.filter(filenames, 'forces.txt'):
                listOfForceFiles.append(os.path.join(dirpath, txtname))

            for txtname in fnmatch.filter(filenames, 'f_ref.txt'):
                listOfInputFiles.append(os.path.join(dirpath, txtname))

        print("Retrieved %d position files, %d force files and %d input files from base directory '%s' "
                         %(len(listOfPositionFiles), len(listOfForceFiles), len(listOfInputFiles), self.base_path))

        # Read the appropriate columns and parse them into np arrays
        for file_idx, filename in enumerate(listOfForceFiles):
            time, seq, f_x, f_y, f_z, t_x, t_y, t_z = np.loadtxt(filename, usecols = (0, 1, 4, 5, 6, 7, 8, 9), skiprows = 1, delimiter = ',', unpack = True)
            f_x = f_x.reshape(-1,1)
            f_y = f_y.reshape(-1,1)
            f_z = f_z.reshape(-1,1)
            forces = np.concatenate((f_x, f_y, f_z), axis = 1)
            self.force_data.update({file_idx : forces})
            full_forces = np.append(full_forces, forces, axis = 0)

        full_forces = full_forces[1:,] # Remove the first "trash" line that was created with np.empty

        for file_idx, filename in enumerate(listOfPositionFiles):
            time, seq, p_x, p_y, p_z = np.loadtxt(filename, usecols = (0, 1, 4, 5, 6), skiprows = 1, delimiter = ',', unpack = True)
            p_x = p_x.reshape(-1,1)
            p_y = p_y.reshape(-1,1)
            p_z = p_z.reshape(-1,1)
            positions = np.concatenate((p_x, p_y, p_z), axis = 1)
            self.position_data.update({file_idx : positions})
            full_positions = np.append(full_positions, positions, axis = 0)

        full_positions = full_positions[1:,] # remove the first "trash" line that was created with np.empty

        for file_idx, filename in enumerate(listOfInputFiles):
            time, seq, f_ref_x, f_ref_y, f_ref_z = np.loadtxt(filename, usecols = (0, 1, 4, 5, 6), skiprows = 1, delimiter = ',', unpack = True)
            f_ref_x = f_ref_x.reshape(-1,1)
            f_ref_y = f_ref_y.reshape(-1,1)
            f_ref_z = f_ref_z.reshape(-1,1)
            inputs = np.concatenate((f_ref_x, f_ref_y, f_ref_z), axis = 1)
            self.input_data.update({file_idx : inputs})
            full_inputs = np.append(full_inputs, inputs, axis = 0)

        full_inputs = full_inputs[1:,] # remove the first "trash" line that was created with np.empty

        faulty_files = 0
        for i in range(len(self.position_data)):
            if len(self.position_data[i]) == len(self.force_data[i]) and len(self.force_data[i]) == len(self.input_data[i]):
                if verbose:
                    print(i, self.position_data[i].shape, self.force_data[i].shape, self.input_data[i].shape)
            else:
                faulty_files += 1
                print("In dataset %s there is a size mismatch. Pos : %d F_s %d F_ref %d " %(listOfPositionFiles[i], self.position_data[i].shape[0], self.force_data[i].shape[0], self.input_data[i].shape[0]))
        print("%d files need to be fixed" %faulty_files)
        full = np.concatenate((full_positions, full_forces, full_inputs), axis = 1)

    def createRelativefull(self):
        """ Creates the relative full dataset that we use to fit the normalizer and also
        builds a dictionary of {d_idx, relative_dataset} for future use."""
        combined_dict = utils.reset_dataset(self.position_data, self.force_data, self.input_data)
        relative_full = np.empty((1,9))

        for d_idx, dataset in combined_dict.items():
            relative_dataset = utils.get_relative_dataset(dataset, self.block_size)
            self.relative_datasets.update({d_idx : relative_dataset})
            relative_full = np.append(relative_full, relative_dataset, axis = 0)
        return relative_full[1:,:]

    def getGradCamScalerParams(self):
        ''' Calculates scalers required for full input dataset,
        as well as for GradCam visualization.'''
        # first get the relative_full dataset
        relative_full = self.createRelativefull()

        self.scaler.fit(relative_full)

        # extract the seperate features to calculate gradcam scalers.
        positions_feat = relative_full[:,:3]
        forces_feat = relative_full[:,3:6]
        inputs_feat = relative_full[:,6:]

        positions_min_max = utils.get_feature_abs_max(positions_feat)
        forces_min_max = utils.get_feature_abs_max(forces_feat)
        inputs_min_max = utils.get_feature_abs_max(inputs_feat)
        self.gradcam_scaler_params = [positions_min_max, forces_min_max, inputs_min_max]

    def createIdxToRunBlockDict(self):
        '''
        Creates a dictionary that translates between full dataset index, and sub-dataset ID + block id
        to be fetched from the CSV importers.
        '''
        dictLengths = [int(len(self.relative_datasets[dk])/self.block_size)-self.block_horizon for dk in self.relative_datasets.keys()]
        totalLength = np.sum(dictLengths)

        currDictIdx = 0
        accrued = 0
        for i in range(totalLength):
            if( i>= dictLengths[currDictIdx]+accrued):
                accrued += dictLengths[currDictIdx]
                currDictIdx += 1
            self.idxToRunDict.update({i:(currDictIdx,i-accrued)})

    def getFrame(self, frame_idx):
        '''
        Returns the input image and class label for a given dataset index
        '''
        frame_label = 0
        d_idx, b_idx = self.idxToRunDict[frame_idx]
        current_block_frame = utils.get_block(self.relative_datasets[d_idx], b_idx, self.block_size)
        current_block_frame_scaled = self.scaler.transform(current_block_frame)

        imgOut = imgutils.renderImage(current_block_frame_scaled, self.cell_size, self.gradcam_scaler_params,self.colormap)

        #move channel axis first
        imgOut=imgOut.squeeze()
        imgOut=np.moveaxis(imgOut,-1,0)

        frame_label = self.getLabel(frame_idx)
        return imgOut, frame_label

    def getLabel(self, frame_idx):
        """ In the binary class problem we determine the label by comparing the UNSCALED
        relative displacement at the last timestep of a block self.block_horizon ahead with
        self.label threshold."""
        d_idx, b_idx = self.idxToRunDict[frame_idx]

        if not self.multi_class:
            frame_label = 0
            next_block_frame = utils.get_block(self.relative_datasets[d_idx], b_idx+self.block_horizon, self.block_size)
            if np.abs(next_block_frame[-1,2]) <= self.label_threshold:
                frame_label = 1

        """ For the multi-class problem, every frame's label will be a one-hot vector x
        where x_i (1 < i <= self.block_horizon) is 1 if block i indicates stuck.
        We add an extra element x[0] = 1 that will only be used to denote that
        none of the blocks within the horizon had a stuckage."""
        if self.multi_class:
            frame_label_onehot = np.zeros((self.block_horizon + 1, 1))
            frame_label_onehot[0] = 1
            for h_idx in range(1, self.block_horizon + 1):
                next_block_frame = utils.get_block(self.relative_datasets[d_idx], b_idx + h_idx, self.block_size)
                if np.abs(next_block_frame[-1,2]) <= self.label_threshold:
                    frame_label_onehot[h_idx] = 1
                    frame_label_onehot[0] = 0
                    break
            frame_label = np.argmax(frame_label_onehot)
        return frame_label

    def scaleGradCamFrame(self, frame, feat_range = (0, 1)):
        frame_copy = np.copy(frame)
        frame_pos = frame_copy[:,:3]
        frame_for = frame_copy[:,3:6]
        frame_in = frame_copy[:,6:]

        pos_data_std = (frame_pos - self.gradcam_scaler_params[0][0]*np.ones_like(frame_pos))/(self.gradcam_scaler_params[0][1]-self.gradcam_scaler_params[0][0])
        scaled_pos = pos_data_std*(max(feat_range) - min(feat_range)) + min(feat_range)

        for_data_std = (frame_for - self.gradcam_scaler_params[1][0]*np.ones_like(frame_for))/(self.gradcam_scaler_params[1][1]-self.gradcam_scaler_params[1][0])
        scaled_for = for_data_std*(max(feat_range) - min(feat_range)) + min(feat_range)

        in_data_std = (frame_in - self.gradcam_scaler_params[2][0]*np.ones_like(frame_in))/(self.gradcam_scaler_params[2][1]-self.gradcam_scaler_params[2][0])
        scaled_in = in_data_std*(max(feat_range) - min(feat_range)) + min(feat_range)

        scaled_frame = np.concatenate((scaled_pos, scaled_for, scaled_in), axis = 1)
        return scaled_frame

    def getVisIntFrame(self, frame_idx):
        '''
        Scales an image according to the visually intuitive scalers used for interpretation.
        '''
        d_idx, b_idx = self.idxToRunDict[frame_idx]

        current_block_frame = utils.get_block(self.relative_datasets[d_idx], b_idx, self.block_size)
        current_block_frame_scaled = self.scaleGradCamFrame(current_block_frame)

        imgOut = imgutils.renderImage(current_block_frame_scaled, self.cell_size, self.gradcam_scaler_params,self.colormap)

        #move channel axis first
        imgOut=imgOut.squeeze()
        imgOut=np.moveaxis(imgOut,-1,0)

        return imgOut
