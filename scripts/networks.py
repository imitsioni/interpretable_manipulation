import torch
import torch.nn as nn
import warnings
import torch.nn.functional as F
from torch.autograd import Variable
#from convolution_lstm import ConvLSTM
#import convolution_lstm.ConvLSTM

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
        if(len(self.channels)>1):
            self.conv2 = nn.Conv2d(channels[0], channels[1], filterSizes[1],\
                                   stride=strides[1], padding=(paddings[1]), bias=biases)

        self.nonlin1 = nn.LeakyReLU()
        if(len(self.channels)>2):
            self.conv3 = nn.Conv2d(channels[1], channels[2], filterSizes[2],\
                                   stride=strides[2], padding=(paddings[2]), bias=biases)

        #calculate final shape of the input for the fully connected layer
        self.h=int(((self.inputImgSize[0]+(2*paddings[0][0]) - (self.filterSizes[0][0]-1) -1)/self.strides[0][0])+1)
        if(len(self.channels)>1):
            self.h=int(((self.h+(2*paddings[1][0]) - (self.filterSizes[1][0]-1) -1)/self.strides[1][0])+1)
        if(len(self.channels)>2):
            self.h=int(((self.h+(2*paddings[2][0]) - (self.filterSizes[2][0]-1) -1)/self.strides[2][0])+1)

        self.w=int(((self.inputImgSize[1]+(2*paddings[0][1]) - (self.filterSizes[0][1]-1) -1)/self.strides[0][1])+1)
        if(len(self.channels)>1):
            self.w=int(((self.w+(2*paddings[1][1]) - (self.filterSizes[1][1]-1) -1)/self.strides[1][1])+1)
        if(len(self.channels)>2):
            self.w=int(((self.w+(2*paddings[2][1]) - (self.filterSizes[2][1]-1) -1)/self.strides[2][1])+1)

        self.fc1 = nn.Linear(self.channels[-1]*self.h*self.w, self.classes)
        self.softmax = torch.nn.Softmax(dim = 1)

    def forward(self, x, train = True):
        x = self.conv1(x)
        if(len(self.channels)>1):
            x = self.conv2(x)

        x = self.nonlin1(x)
        if(len(self.channels)>2):
            x = self.conv3(x)
        x = self.fc1(x.view(-1,self.channels[-1]*self.h*self.w))

        if (self.training):
            out = x
        else:
            out = self.softmax(x)
        return out
    
class EventNetFC(nn.Module):
    def __init__(self, inputImgSize,LayerSizes=[90,90,90],classes=2, channels_in=1): 
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
        super(EventNetFC, self).__init__()
        self.inputImgSize=inputImgSize
        self.LayerSizes=LayerSizes
        self.channels_in=channels_in
        self.fc1 = nn.Linear(inputImgSize[0]*inputImgSize[1]*channels_in, LayerSizes[1])
        self.fc2 = nn.Linear(LayerSizes[1], LayerSizes[2])
        self.fc3 = nn.Linear(LayerSizes[2],classes)
        
        if classes>2: warnings.warn("Multiclass problem, make sure you have the correct dataloader.")
        self.classes = classes

        self.softmax = torch.nn.Softmax(dim = 1)

    def forward(self, x, train = True):
        x = x.view(-1,self.inputImgSize[0]*self.inputImgSize[1]*self.channels_in)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc2(x)
        
        if (self.training):
            out = x
        else:
            out = self.softmax(x)
        return out
    
#---------------------------------------------------------------
class EventNetLSTM(nn.Module):
    def __init__(self, inputImgSize,LayerSizes=[9],classes=2, channels_in=1,layers=3): 
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
        super(EventNetLSTM, self).__init__()
        self.layerSizes = LayerSizes*channels_in
        self.channels_in=channels_in
        self.layers = layers
        self.inputImgSize=inputImgSize
        self.lstm = nn.LSTM(input_size = inputImgSize[0]*channels_in,\
                            hidden_size=self.layerSizes[0],num_layers=self.layers)
        self.fc = nn.Linear(self.layerSizes[-1],classes)
        
        
        if classes>2: warnings.warn("Multiclass problem, make sure you have the correct dataloader.")
        self.classes = classes

        self.softmax = torch.nn.Softmax(dim = 1)

    def forward(self, x, train = True):
        self.hidden = self.initHidden(x.shape[0])

        x = x.view(-1,self.inputImgSize[1],self.inputImgSize[0]*self.channels_in)
        x= x.permute(1,0,2)
     
        hiddens, x =self.lstm(x,self.hidden)
        x=x[0][-1]

        
        x = self.fc(x)
        
        if (self.training):
            out = x
        else:
            out = self.softmax(x)

        return out
    
    def initHidden(self,batch_size):
        hidden_state = torch.randn(self.layers, batch_size, self.layerSizes[0]).float().cuda()
        cell_state = torch.randn(self.layers, batch_size, self.layerSizes[0]).float().cuda()
        return (hidden_state, cell_state)

"""
Based on implementation from https://github.com/automan000/Convolution_LSTM_pytorch
"""

class CLSTMEventNet(torch.nn.Module):
    def __init__(self, num_classes=174, nb_lstm_units=32, channels=3, conv_kernel_size=(5, 5), pool_kernel_size=(2, 2),
                 top_layer=True, avg_pool=False, max_pool=False,batch_normalization=True, lstm_layers=4, step=16,
                 image_size=(224, 224), dropout=0, conv_stride=(1, 1), effective_step=[4, 8, 12, 15],
                 use_entire_seq=False, add_softmax=True,device="cpu"):

        super(CLSTMEventNet, self).__init__()

        self.num_classes = num_classes
        self.nb_lstm_units = nb_lstm_units
        self.channels = channels
        self.top_layer = top_layer
        self.avg_pool = avg_pool
        self.max_pool = max_pool
        self.c_kernel_size = conv_kernel_size
        self.lstm_layers = lstm_layers
        self.step = step
        self.im_size = image_size
        self.pool_kernel_size = pool_kernel_size
        self.batch_normalization = batch_normalization
        self.dropout = dropout
        self.conv_stride = conv_stride
        self.effective_step = effective_step
        self.add_softmax = add_softmax
        self.use_entire_seq = use_entire_seq
        self.clstm = None
        self.endFC = None
        self.sm = None
        self.device = device
        
        if(self.max_pool==False):
            self.pool_kernel_size=(1,1)

        self.build()

    def build(self):
        """
        pytorch CLSTM usage: 
        clstm = ConvLSTM(input_channels=512, hidden_channels=[128, 64, 64],
                         kernel_size=5, step=9, effective_step=[2, 4, 8])
        lstm_outputs = clstm(cnn_features)
        hidden_states = lstm_outputs[0]
        """
        self.clstm = ConvLSTM(input_channels=self.channels,
                              hidden_channels=[self.nb_lstm_units] * self.lstm_layers,
                              kernel_size=self.c_kernel_size[0], conv_stride=self.conv_stride,
                              pool_kernel_size=self.pool_kernel_size, max_pool=self.max_pool, step=self.step,
                              effective_step=self.effective_step, device = self.device,
                              batch_normalization=self.batch_normalization, dropout=self.dropout)

        if self.use_entire_seq:
            self.endFC = torch.nn.Linear(in_features=len(self.effective_step) * self.nb_lstm_units * int(
                self.im_size[0] / ((self.conv_stride * self.pool_kernel_size[0]) ** self.lstm_layers)) * int(
                self.im_size[1] / ((self.conv_stride * self.pool_kernel_size[0]) ** self.lstm_layers)),
                                         out_features=self.num_classes)
        else:
            self.endFC = torch.nn.Linear(in_features=self.nb_lstm_units * int(
                self.im_size[0] / ((self.conv_stride * self.pool_kernel_size[0]) ** self.lstm_layers)) * int(
                self.im_size[1] / ((self.conv_stride * self.pool_kernel_size[0]) ** self.lstm_layers)),
                                         out_features=self.num_classes)

        print("use entire sequence is: ", self.use_entire_seq)
        print("shape of FC is: ", self.endFC)
        self.sm = torch.nn.Softmax(dim=1)

    def forward(self, x):

        output, hiddens = self.clstm(x)

        if self.use_entire_seq:
            output = self.endFC(torch.stack(output).view(-1, len(self.effective_step) * self.nb_lstm_units * int(
                self.im_size[0] / ((self.conv_stride * self.pool_kernel_size[0]) ** self.lstm_layers)) * int(
                self.im_size[1] / ((self.conv_stride * self.pool_kernel_size[0]) ** self.lstm_layers))))
        else:
            output = self.endFC(output[-1].view(-1, self.nb_lstm_units * int(
                self.im_size[0] / ((self.conv_stride * self.pool_kernel_size[0]) ** self.lstm_layers)) * int(
                self.im_size[1] / ((self.conv_stride * self.pool_kernel_size[0]) ** self.lstm_layers))))

        if self.add_softmax:
            output = self.sm(output)

        return output

    def extract_features(self, x):
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x)
        return self.avg_pool(x)
    
    
    
class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, conv_stride, device):
        super(ConvLSTMCell, self).__init__()

        assert hidden_channels % 2 == 0

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_features = 4
        self.conv_stride = conv_stride
        self.device = device

        self.padding = int((kernel_size - 1) / 2)

        self.Wxi = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, self.conv_stride, self.padding, bias=True)
        self.Whi = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxf = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, self.conv_stride, self.padding, bias=True)
        self.Whf = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxc = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, self.conv_stride, self.padding, bias=True)
        self.Whc = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxo = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, self.conv_stride, self.padding, bias=True)
        self.Who = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)

        self.Wci = None
        self.Wcf = None
        self.Wco = None

    def forward(self, x, h, c):
        #print("x shape ", x.shape, " h shape: ", h.shape, " c shape: ", c.shape)
        #print("wci shape: ",self.Wxi(x).shape)
        #print("whi shape: ", self.Whi(x).shape)
        #print("self wci op shape: ",(c * self.Wci).shape)
        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c * self.Wci)
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h) + c * self.Wcf)
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h) + cc * self.Wco)
        ch = co * torch.tanh(cc)
        return ch, cc

    def init_hidden(self, batch_size, hidden, shape):
        if self.Wci is None:
            self.Wci = Variable(torch.zeros(1, hidden, shape[0]//self.conv_stride, shape[1]//self.conv_stride)).to(self.device)
            self.Wcf = Variable(torch.zeros(1, hidden, shape[0]//self.conv_stride, shape[1]//self.conv_stride)).to(self.device)
            self.Wco = Variable(torch.zeros(1, hidden, shape[0]//self.conv_stride, shape[1]//self.conv_stride)).to(self.device)
        else:
            assert shape[0]//self.conv_stride == self.Wci.size()[2], 'Input Height Mismatched! %d vs %d' %(shape[0]//self.conv_stride, self.Wci.size()[2])
            assert shape[1]//self.conv_stride == self.Wci.size()[3], 'Input Width Mismatched!'
        #print("returning init h of size ", batch_size, hidden, shape[0], shape[1])
        return (Variable(torch.zeros(batch_size, hidden, shape[0]//self.conv_stride, shape[1]//self.conv_stride)).to(self.device),
                Variable(torch.zeros(batch_size, hidden, shape[0]//self.conv_stride, shape[1]//self.conv_stride)).to(self.device))


class ConvLSTM(nn.Module):
    # input_channels corresponds to the first input feature map
    # hidden state is a list of succeeding lstm layers.
    def __init__(self, input_channels, hidden_channels, kernel_size, conv_stride,
                 pool_kernel_size=(2,2), max_pool=False, step=1, effective_step=[1],
                 batch_normalization=True, dropout=0, device='cpu'):
        super(ConvLSTM, self).__init__()
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)
        self.step = step
        self.effective_step = effective_step
        self._all_layers = []
        self.pool_kernel_size = pool_kernel_size
        self.max_pool = max_pool
        self.conv_stride = conv_stride
        self.mp = nn.MaxPool2d(kernel_size=self.pool_kernel_size)
        self.batch_norm = batch_normalization
        self.dropout_rate=dropout
        self.device = device
        #to be pool_size=2, strides=None, padding='valid', data_format='channels_last')
        #kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        self.bn = nn.BatchNorm2d(self.hidden_channels[0], eps=1e-05, momentum=0.1, affine=True) #should prolly be several, and in loop with cell{] thing
        #name = 'bn'
        self.dropout = torch.nn.Dropout(p=self.dropout_rate)
        for i in range(self.num_layers):
            name = 'cell{}'.format(i)
            cell = ConvLSTMCell(self.input_channels[i], self.hidden_channels[i], self.kernel_size, self.conv_stride, self.device)
            setattr(self, name, cell)
           
            
            self._all_layers.append(cell)

    def forward(self, input):
        internal_state = []
        outputs = []
        for step in range(self.step):
            x = input[:,:,step,:,:]
            for i in range(self.num_layers):
                # all cells are initialized in the first step
                name = 'cell{}'.format(i)
                #print("on cell step ", i ,"with name: ",name)
                #print("x size is:", x.size())
                if step == 0:
                    bsize, channels, height, width = x.size()
                    (h, c) = getattr(self, name).init_hidden(batch_size=bsize, hidden=self.hidden_channels[i],
                                                             shape=(height,width))
                                                             #shape=(int(height/max(1,(self.pool_kernel_size[0]*(i)))),\
                                                             #       int(width/max(1,(self.pool_kernel_size[1]*(i))))))
                    internal_state.append((h, c))

                # do forward
                (h, c) = internal_state[i]
                #print("h size is: ", h.size())
                x, new_c = getattr(self, name)(x, h, c)
                internal_state[i] = (x, new_c)
                
                if(self.dropout_rate):
                    x = self.dropout(x)
                if(self.batch_norm):
                    x = self.bn(x)
                if(self.max_pool):
                    x = self.mp(x)
                
                
            # only record effective steps
            if step in self.effective_step:
                outputs.append(x)

        #print("returning shape: ", len(outputs), outputs[-1].shape)
        return outputs, (x, new_c)