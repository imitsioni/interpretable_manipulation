# Code Structure and Details
- The code for training and evaluation is in `scripts/`.
- Datasets from the cutting task are placed in `data/cutting_datasets`.

## Installing dependencies  
```bash
virtualenv interp_event_detection_env --python=python3 # Or use preferred env creation method
cd interpretable_manip_env && source bin/activate # Activate env
pip3 install numpy matplotlib scikit-learn joblib jupyter torch torchvision opencv-python # Dependencies
```

## Training
All models were trained to minimize Cross Entropy and use Adam as an optimizer.
To train a model to be used for evaluation, run the script `train.py`. Usage is as follows:

`train.py --modelName` {file name to save model as}

Further arguments can be given to overwrite the default values that were used in the presented work:

    --train_path : path to training data (default  ../data/cutting_datasets/train)

    --test_path : path to test data (default ../data/cutting_datasets/test),

    --horizon : How many blocks ahead class label refers to (default 3)

    --block_size : 1Length M of the sequence comprising each block. (default 10)

    --scale_pixels : upscaling factor for input representation (default 1)

    --colormap : colormap to be used for input rendering (default "seismic")

    --epochs : epochs to run training (default 25)

    --learning_rate : initial learning rate (default 1e-03)

    --batch_size : batch size used for training (default 32)

    --num_workers : number of worker threads used for training (default 2)

    --model_name : file name to save model as, used to load later for evaluation

## Evaluation
To view the visually interperatable outputs, and conduct the feature importance evaluation, run the `Visualizing.ipynb` notebook. The same default arguments as above can be changed in the first cell, as well as where and if figures should be saved.


# Additional Notes
## Colormap choice
The colormap choice can affect the training results for the CNN and needs to be chosen accordingly. For a task with specific directionality, e.g. the pushing task that transpires only on one motion axis and all the quantities can have the same sign, sequential colormaps are appropriate.
However, in other cases, the features can take both positive and negative values that lead to different dynamic patterns, making diverging colormaps more fitting.
During training, we observed that the color chosen to represent the neutral zone of the scaled features can bias the network and lead to significantly different results.
## Data collection
During data collection the commanded trajectories always ended with zero velocities for stability issues. For this specific problem formulation, these sequences were kept in the dataset and unavoidably annotated as part of the negative class C = 1. Depending on the actual application of the event detection module, this is not always advisable. In the case that the event triggers a reaction from the system, these points should be annotated differently, e.g. with an additional clause for the control input level that will differentiate them.
