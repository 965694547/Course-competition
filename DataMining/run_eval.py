import torch
import torch.nn as nn
import math
import pandas as pd
import os
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import random_split

epoch = 1
path = 'model_'+ str(epoch) + '.pkl'

'''
input_post = pd.read_csv('testpost.csv',index_col=0,header=None)
input_post_out = pd.read_csv('testpost.csv').values
input_after = pd.read_csv('testafter.csv',index_col=0,header=None)
input_after_out = pd.read_csv('testafter.csv').values
x = np.vstack((input_post_out[:,1:],input_after_out[:,1:]))
np.savetxt("x_test_ok.txt", x, fmt="%.5f", delimiter=",")
'''

x = np.loadtxt("x_test_ok.txt", dtype=float, delimiter=",")

class DataTensor(Dataset):
    def __init__(self, train_tensor, label_tensor):
        self.train_tensor = train_tensor
        self.label_tensor = label_tensor

    def __getitem__(self, index):
        return self.train_tensor[index], self.label_tensor[index]

    def __len__(self):
        return self.train_tensor.size(0)

def get_data_genertor(train_data):
    label_tensor = torch.tensor([label[1] for label in train_data])
    train_tensor = [train[0] for train in train_data]
    train_tensor = torch.cat(train_tensor)

    return train_tensor, label_tensor

class ConvNormAct(nn.Module):
    '''
    This class defines the convolution layer with normalization and a PReLU
    activation
    '''

    def __init__(self, nIn, nOut, kSize, stride=1, groups=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        '''
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv1d(nIn, nOut, kSize, stride=stride, padding=padding,
                              bias=True, groups=groups)
        self.norm = nn.GroupNorm(1, nOut, eps=1e-08)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        output = self.conv(input)
        output = self.norm(output)
        return self.act(output)


class ConvNorm(nn.Module):
    '''
    This class defines the convolution layer with normalization and PReLU activation
    '''

    def __init__(self, nIn, nOut, kSize, stride=1, groups=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        '''
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv1d(nIn, nOut, kSize, stride=stride, padding=padding,
                              bias=True, groups=groups)
        self.norm = nn.GroupNorm(1, nOut, eps=1e-08)

    def forward(self, input):
        output = self.conv(input)
        return self.norm(output)


class NormAct(nn.Module):
    '''
    This class defines a normalization and PReLU activation
    '''
    def __init__(self, nOut):
        '''
        :param nOut: number of output channels
        '''
        super().__init__()
        self.norm = nn.GroupNorm(1, nOut, eps=1e-08)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        output = self.norm(input)
        return self.act(output)


class DilatedConv(nn.Module):
    '''
    This class defines the dilated convolution.
    '''

    def __init__(self, nIn, nOut, kSize, stride=1, d=1, groups=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        '''
        super().__init__()
        self.conv = nn.Conv1d(nIn, nOut, kSize, stride=stride, dilation=d,
                              padding=((kSize - 1) // 2) * d, groups=groups)

    def forward(self, input):
        return self.conv(input)


class DilatedConvNorm(nn.Module):
    '''
    This class defines the dilated convolution with normalized output.
    '''

    def __init__(self, nIn, nOut, kSize, stride=1, d=1, groups=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        '''
        super().__init__()
        self.conv = nn.Conv1d(nIn, nOut, kSize, stride=stride, dilation=d,
                              padding=((kSize - 1) // 2) * d, groups=groups)
        self.norm = nn.GroupNorm(1, nOut, eps=1e-08)

    def forward(self, input):
        output = self.conv(input)
        return self.norm(output)


class block(nn.Module):
    '''
    This class defines the Upsampling block, which is based on the following
    principle:
        REDUCE ---> SPLIT ---> TRANSFORM --> MERGE
    '''

    def __init__(self,
                 out_channels=128,
                 in_channels=512,
                 upsampling_depth=4):
        super().__init__()
        self.proj_1x1 = ConvNormAct(out_channels, in_channels, 1,
                                    stride=1, groups=1)
        self.depth = upsampling_depth
        self.spp_dw = nn.ModuleList()
        self.spp_dw.append(DilatedConvNorm(in_channels, in_channels, kSize=5,
                                           stride=1, groups=in_channels, d=1))

        for i in range(1, upsampling_depth):
            if i == 0:
                stride = 1
            else:
                stride = 2
            self.spp_dw.append(DilatedConvNorm(in_channels, in_channels,
                                               kSize=2*stride + 1,
                                               stride=stride,
                                               groups=in_channels, d=1))
        if upsampling_depth > 1:
            self.upsampler = torch.nn.Upsample(scale_factor=2,
                                               # align_corners=True,
                                               # mode='bicubic'
                                               )
        self.conv_1x1_exp = ConvNorm(in_channels, out_channels, 1, 1, groups=1)
        self.final_norm = NormAct(in_channels)
        self.module_act = NormAct(out_channels)

    def forward(self, x):
        '''
        :param x: input feature map
        :return: transformed feature map
        '''

        # Reduce --> project high-dimensional feature maps to low-dimensional space
        output1 = self.proj_1x1(x)
        output = [self.spp_dw[0](output1)]

        # Do the downsampling process from the previous level
        for k in range(1, self.depth):
            out_k = self.spp_dw[k](output[-1])
            output.append(out_k)

        # Gather them now in reverse order
        for _ in range(self.depth-1):
            resampled_out_k = self.upsampler(output.pop(-1))
            output[-1] = output[-1] + resampled_out_k

        expanded = self.conv_1x1_exp(self.final_norm(output[-1]))

        return self.module_act(expanded + x)


class anti(nn.Module):
    def __init__(self,
                 out_channels=128,
                 in_channels=512,
                 num_blocks=16,
                 upsampling_depth=4,
                 enc_kernel_size=21,
                 enc_num_basis=512,
                 ):
        super(anti, self).__init__()

        # Number of sources to produce
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.upsampling_depth = upsampling_depth
        self.enc_kernel_size = enc_kernel_size
        self.enc_num_basis = enc_num_basis


        # Appropriate padding is needed for arbitrary lengths
        self.lcm = abs(self.enc_kernel_size // 2 * 2 **
                       self.upsampling_depth) // math.gcd(
                       self.enc_kernel_size // 2,
                       2 ** self.upsampling_depth)

        # Front end
        self.conv1 = nn.Sequential(*[
            nn.Conv1d(in_channels=1, out_channels=enc_num_basis,
                      kernel_size=enc_kernel_size,
                      stride=enc_kernel_size // 2,
                      padding=enc_kernel_size // 2),
            nn.ReLU(),
        ])

        # Norm before the rest, and apply one more dense layer
        self.ln = nn.GroupNorm(1, enc_num_basis, eps=1e-08)
        self.l1 = nn.Conv1d(in_channels=enc_num_basis,
                            out_channels=out_channels,
                            kernel_size=1)

        # Separation module
        self.sm = nn.Sequential(*[
            block(out_channels=out_channels,
                   in_channels=in_channels,
                   upsampling_depth=upsampling_depth)
            for r in range(num_blocks)])

        self.max_pool1 = nn.MaxPool2d(3, 2)

        self.fc1 = nn.Sequential(
            nn.Linear(1701,567),
            nn.ReLU(True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(567,189),
            nn.ReLU(True)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(189,63),
            nn.ReLU(True)
        )
        self.fc4 = nn.Sequential(
            nn.Linear(63, class_num),
            nn.ReLU(True)
        )

    # Forward pass
    def forward(self, input_wav):
        # Front end
        x = self.pad_to_appropriate_length(input_wav)
        x = self.conv1(x)

        # Separation module
        x = self.ln(x)
        x = self.l1(x)
        x = self.sm(x)#3*128*56

        x = self.max_pool1(x)  # 3*63*27
        x = torch.flatten(x, 1)
        #x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        if class_num == 4  or class_num == 5:
            y = torch.sigmoid(x.to(dtype=torch.float64))
        else:
            y = torch.softmax(x.to(dtype=torch.float64))
        return y

    def pad_to_appropriate_length(self, x):
        values_to_pad = int(x.shape[-1]) % self.lcm
        if values_to_pad:
            appropriate_shape = x.shape
            padded_x = torch.zeros(
                list(appropriate_shape[:-1]) +
                [appropriate_shape[-1] + self.lcm - values_to_pad],
                dtype=torch.float32)
            padded_x[..., :x.shape[-1]] = x
            return padded_x
        return x

    @staticmethod
    def remove_trailing_zeros(padded_x, initial_x):
        return padded_x[..., :initial_x.shape[-1]]

if __name__ == '__main__':
    class_num = 5
    batch_size = 3
    net = anti(out_channels=128,
                     in_channels=512,
                     num_blocks=16,
                     upsampling_depth=4,
                     enc_kernel_size=21,
                     enc_num_basis=512,
                    )
    net.load_state_dict(torch.load(path))
    net.eval()
    input_post = pd.read_csv('testpost.csv',header=None).values
    input_after = pd.read_csv('testafter.csv',header=None).values
    time = np.hstack((input_post[:,0],input_after[:,0]))
    # output_result = pd.DataFrame(columns=["Time","CST","IRF","SRF","PED","HRF"])
    output_result = []
    with torch.no_grad():
        for i in range(0,len(time)-3,batch_size):
            output_result.append(
                                net(torch.reshape(torch.from_numpy(x[i:i+3,:]), (batch_size, 1, -1))).numpy()
                                )
            print(i)
    output_result  = np.array(output_result).reshape(-1,5)
    df = pd.DataFrame(output_result)
    df.to_csv('result.csv', header=0, index=0)




