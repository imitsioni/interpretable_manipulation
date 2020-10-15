import torch
import torch.nn as nn
import warnings
import torch.nn.functional as F

class MultiEventNetTemplate(nn.Module):
    def __init__(self, inputImgSize,channels,filterSizes,strides,paddings, biases=True,classes=2, channels_in=1): 
        '''
        arguments:
            inputImgSize: The dimensions of the input representation, given in height, width
            channels: Amount of channels (filters) to be used for each conv layer, starting with the first
            filterSizes: The filter sizes (in tuples) to be used for each conv layer, starting with the first
            paddings: Padding to be used for each conv layer, starting with the first
            biases: If biases should be used for conv layers
            classes: Number of classes the task has
            channels_in: Number of input channels the input representation has
        '''
        super(MultiEventNetTemplate, self).__init__()

        if classes>2: warnings.warn("Multiclass problem, make sure you have the correct dataloader.")
        self.classes = classes
        self.channels = channels
        self.strides=strides
        self.filterSizes=filterSizes
        self.paddings=paddings
        self.inputImgSize = inputImgSize
        self.conv1 = nn.Conv2d(channels_in, channels[0], filterSizes[0], stride=strides[0], padding=(paddings[0]), bias=biases)
        self.conv2 = nn.Conv2d(channels[0], channels[1], filterSizes[1], stride=strides[1], padding=(paddings[1]), bias=biases)

        self.nonlin1 = nn.LeakyReLU()

        self.conv3 = nn.Conv2d(channels[1], channels[2], filterSizes[2], stride=strides[2], padding=(paddings[2]), bias=biases)

        #calculate final shape of the input for the fully connected layer
        self.h1=int(((self.inputImgSize[0]+(2*paddings[0][0]) - (self.filterSizes[0][0]-1) -1)/self.strides[0][0])+1)
        self.h2=int(((self.h1+(2*paddings[1][0]) - (self.filterSizes[1][0]-1) -1)/self.strides[1][0])+1)
        self.h3=int(((self.h2+(2*paddings[2][0]) - (self.filterSizes[2][0]-1) -1)/self.strides[2][0])+1)

        self.w1=int(((self.inputImgSize[1]+(2*paddings[0][1]) - (self.filterSizes[0][1]-1) -1)/self.strides[0][1])+1)
        self.w2=int(((self.w1+(2*paddings[1][1]) - (self.filterSizes[1][1]-1) -1)/self.strides[1][1])+1)
        self.w3=int(((self.w2+(2*paddings[2][1]) - (self.filterSizes[2][1]-1) -1)/self.strides[2][1])+1)

        self.fc1 = nn.Linear(self.channels[-1]*self.h3*self.w3, self.classes)
        self.softmax = torch.nn.Softmax(dim = 1)

    def forward(self, x, train = True):
        x = self.conv1(x)
        x = self.conv2(x)

        x = self.nonlin1(x)

        x = self.conv3(x)
        x = self.fc1(x.view(-1,self.channels[-1]*self.h3*self.w3))

        if (self.training):
            out = x
        else:
            out = self.softmax(x)
        return out