import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm


def makeGifOfImages(dataFrames,savePath,removeStills=False):
    if(not os.path.exists(savePath)):
        os.mkdir(savePath)

    for idx, frame in enumerate(dataFrames):
        plt.imsave(savePath+"img{}.png".format(idx), frame, cmap = 'gist_gray')

    #create gif
    path_to_combined_gif = os.path.join(savePath, "data.mp4")

    os.system("convert -delay 5 -loop 0 {}.png {}".format(
                            os.path.join(savePath, "img*"),
                            path_to_combined_gif))
    if(removeStills):
        os.system("rm {}/img*".format(savePath))

def renderImage(data, cellSize, scaler_params,colormap):
    '''
    summary: render a given block to an image
    inputs:
        data: a (sample x features) array containing the data to be rendered (for example 10 x 7)
        cellSize: a tuple containing the desired area of a cell in the image.
                  Size (pixels) should be given in (time_width,feature_height)
        colorMap: Which colormap to use for creating the RGP input representation
    returns:
        a numpy array of the rendered image. Can be viewed with plt.imshow()
    '''
    #transpose data to be Feature x Sample (since sample_t should be x axis in IMAGE)
    imgOut = np.transpose(data.copy())

    #increase to area size along time axis
    imgOut = np.repeat(imgOut, cellSize[0],axis=1)
    #increase to area size along feature axis
    imgOut = np.repeat(imgOut, cellSize[1],axis=0)
    
    #create 3 channel input representation, using given colormap
    current_cm = cm.get_cmap(colormap)
    imgOut = current_cm(imgOut)
    imgOut = imgOut[:,:,:3].squeeze()
    
    return imgOut[None,...]
