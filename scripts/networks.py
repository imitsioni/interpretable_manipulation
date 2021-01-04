import torch
import torch.nn as nn
import warnings
import torch.nn.functional as F
from torch.autograd import Variable

class MultiEventNetTemplate(nn.Module):
    def __init__(self, inputImgSize,channels,filterSizes,strides,paddings, biases=True,classes=2, channels_in=1): 
        '''
        CNN model definition used in the manuscript.
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

        # Calculate final shape of the input for the fully connected layer
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

"""
Based on implementation from https://github.com/automan000/Convolution_LSTM_pytorch
"""
class CLSTMEventNet(torch.nn.Module):
    '''
    CLSTM model definition used in the manuscript.
    arguments:
        inputImgSize: The dimensions of the input representation, given in height, width
        classes: Number of classes the task has.
        nb_lstm_units: How many lstm units should be in each CLSTM layer (number of filters). Default 32
        channels_in: Number of input channels the input representation has
        conv_kernel_size: The convolutional filter size used for each layer. Default is (3, 1) in the work.
        pool_kernel_size: Average pooling kernel size.
        avg_pool: Applies average pooling between each layer if True. Is False in the work.
        max_pool: Applies max pooling between each layer if True. Is False in the work.
        batch_normalization: Applies batch norm between each layer if True. Is False in the work.
        lstm_layers: How many CLSTM layers to use in the model. 3 were used in the work.
        step: Input sequence length. Default is 3.
        dropout: If dropout should be applied between each layer. Default is no dropout.
        conv_stride: The stride used in the convolutional operations. Default is (1,1).
        effective_step: Which hidden representations in the sequence is sent to subsequent layers. Default is all.
        add_softmax: If softmax should be applied for class scores. Default is true.
        device: Hardware device to use. 
        
    '''
    def __init__(self, inputImgSize=(9, 10), classes=2, nb_lstm_units=32, channels_in=3, conv_kernel_size=(3, 1),\
                 pool_kernel_size=(2, 2),
                 avg_pool=False, max_pool=False,batch_normalization=False, lstm_layers=3, step=3,
                 dropout=0, conv_stride=(1, 1), effective_step=[10],
                 use_entire_seq=False, add_softmax=True,device="cpu"):

        super(CLSTMEventNet, self).__init__()

        self.classes = classes
        self.nb_lstm_units = nb_lstm_units
        self.channels = channels_in
        self.avg_pool = avg_pool
        self.max_pool = max_pool
        self.c_kernel_size = conv_kernel_size
        self.lstm_layers = lstm_layers
        self.step = step
        self.im_size = inputImgSize
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
                                         out_features=self.classes)
        else:
            self.endFC = torch.nn.Linear(in_features=self.nb_lstm_units * int(
                self.im_size[0] / ((self.conv_stride * self.pool_kernel_size[0]) ** self.lstm_layers)) * int(
                self.im_size[1] / ((self.conv_stride * self.pool_kernel_size[0]) ** self.lstm_layers)),
                                         out_features=self.classes)

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

        self.bn = nn.BatchNorm2d(self.hidden_channels[0], eps=1e-05, momentum=0.1, affine=True) 
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

                if step == 0:
                    bsize, channels, height, width = x.size()
                    (h, c) = getattr(self, name).init_hidden(batch_size=bsize, hidden=self.hidden_channels[i],
                                                             shape=(height,width))
                                                             
                    internal_state.append((h, c))

                # do forward
                (h, c) = internal_state[i]
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

        return outputs, (x, new_c)