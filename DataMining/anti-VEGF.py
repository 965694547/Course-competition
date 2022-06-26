import torch
import torch.nn as nn
import math
import pandas as pd
import os
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import random_split

# path = 'C:\\Users\\GZF\\Desktop\\'

# isExists=os.path.exists(path +'checkpoint')
# if not isExists:
#    os.makedirs(path +'checkpoint')

'''数据处理
file=pd.read_csv('ok.csv',index_col=0)
output_post=file[['preCST','preIRF','preSRF','prePED','preHRF']]
output_after=file[['CST','IRF','SRF','PED','HRF']]

input_post = pd.read_csv('trainpost.csv',index_col=0,header=None)
input_post_out = pd.read_csv('trainpost.csv').values
y1 = np.array(output_post.loc[input_post_out[0,0][0:10]].array.T)

for item in input_post_out[0:,0]:
    try:
        y1 = np.vstack((y1,np.array(output_post.loc[item[0:10]].array.T)))
    except:
        input_post.drop([item], inplace=True)
x1 = input_post.values

input_after = pd.read_csv('trainafter.csv',index_col=0,header=None)
input_after_out = pd.read_csv('trainafter.csv').values
y2 = np.array(output_after.loc[input_after_out[0,0][0:10]].array.T)
for item in input_after_out[0:,0]:
    try:
        y2 = np.vstack((y2,np.array(output_after.loc[item[0:10]].array.T)))
    except:
        input_after.drop([item], inplace=True)
x2 = input_after.values

x = np.vstack((x1,x2))
y = np.vstack((y1,y2))
np.savetxt("x_ok.txt", x, fmt="%.5f", delimiter=",")
np.savetxt("y_ok.txt", y, fmt="%.5f", delimiter=",")
x = np.loadtxt("x_ok.txt", dtype=float, delimiter=",")
y = np.loadtxt("y_ok.txt", dtype=float, delimiter=",")
'''

'''
#数据处理2
file=pd.read_csv('ok.csv',index_col=0)
output_post=file[['preCST','preIRF','preSRF','prePED','preHRF']]
output_after=file[['CST','IRF','SRF','PED','HRF']]
tag_6=pd.read_csv('tag_6.csv',index_col=None,header=None)
input_post = pd.read_csv('trainpost.csv',index_col=0,header=None)
input_post_out = pd.read_csv('trainpost.csv').values
y1 = np.array(output_post.loc[tag_6.values[1,0][0:10]].array.T)
x_6_1 = np.array(input_post.loc[tag_6.values[1,0][0:10]])
for item in tag_6.values[1:,0]:
    try:
        if item != 'nan':
            y1 = np.vstack((y1, np.array(output_post.loc[item[0:10]].array.T)))
            x_6_1 = np.vstack((x_6_1,input_post.loc[item[0:10]]))
    except:
        input_post.drop([item], inplace=True)

input_after = pd.read_csv('trainafter.csv',index_col=0,header=None)
input_after_out = pd.read_csv('trainafter.csv').values
y2 = np.array(output_after.loc[tag_6.values[1,1][0:10]].array.T)
x_6_2 = np.array(input_after.loc[tag_6.values[1,1][0:10]])
for item in tag_6.values[1:,1]:
    try:
        if item != 'nan':
            y2 = np.vstack((y2, np.array(output_after.loc[item[0:10]].array.T)))
            x_6_2 = np.vstack((x_6_2,input_after.loc[item[0:10]]))
    except:
        input_after.drop([item], inplace=True)
x = np.vstack((x_6_1,x_6_2))
y = np.vstack((y1,y2))
np.savetxt("x_ok_6.txt", x, fmt="%.5f", delimiter=",")
np.savetxt("y_ok_6.txt", y, fmt="%.5f", delimiter=",")
'''
x = np.loadtxt("x_ok_6.txt", dtype=float, delimiter=",")
y = np.loadtxt("y_ok_6.txt", dtype=float, delimiter=",")

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

        self.dense0 = nn.Sequential(
            nn.Linear(in_channels*6, in_channels*2),
        )
        self.dense1 = nn.Sequential(
            nn.Linear(in_channels*2, in_channels*1),
            nn.ReLU(True)
        )
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
            nn.Linear(155,55),
            #nn.ReLU(True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(55,15),
            #nn.ReLU(True)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(15,5),
            nn.ReLU(True)
        )
        self.fc4 = nn.Sequential(
            nn.Linear(63, class_num),
            nn.ReLU(True)
        )

        #2.0
        self.lstm = nn.LSTM(input_size = 512,
                       hidden_size = 256,
                       num_layers = 3,
                       batch_first=True, bidirectional=True)
        self.conv2 = nn.Sequential(*[
            nn.Conv1d(in_channels=6, out_channels=256,
                      kernel_size=enc_kernel_size,
                      stride=enc_kernel_size // 2,
                      padding=enc_kernel_size // 2),
            nn.ReLU(),
        ])
        self.max_pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv1d(in_channels=128,
                            out_channels=64,
                            kernel_size=4,
                            stride=4// 2)
        self.max_pool3 = nn.MaxPool2d(4, 2)

    # Forward pass
    def forward(self, input_wav):
        #x = self.dense0(input_wav)
        #x = self.dense1(x)#3,6,512

        #2.0
        x,_ = self.lstm(input_wav)#3,6,512
        x = self.conv2(x)#3,256,52
        x = self.max_pool2(x)#3,128,26
        x = self.conv3(x)#3,64,12
        x = self.max_pool3(x)#3,31,5

        # Front end
        #x = self.pad_to_appropriate_length(x)
        #x = self.conv1(x)

        # Separation module
        #x = self.ln(x)
        #x = self.l1(x)
        #x = self.sm(x)#3*128*56

        #x = self.max_pool1(x)  # 3*63*27
        x = torch.flatten(x, 1)
        #x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        #x = self.fc4(x)
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

    # 超参数
    # 输出类别 为1的时候为回归 为4的时候为分类
    class_num = 5
    # 训练批次
    batch_size = 2
    # 定义超参数 (训练周期)
    epochs = 200
    # 训练长度
    length = len(y)

    #数据处理
    train_data = torch.from_numpy(x.reshape(length,-1))
    label_data = torch.arange(0,length)

    train_dataset = DataTensor(train_data, label_data)
    len_train = int(len(train_dataset) * 0.9)
    train_data, valid_data = random_split(train_dataset, [len_train, len(train_dataset) - len_train])#训练集验证集分类

    #模型设置
    model = anti(out_channels=128,
                     in_channels=512,
                     num_blocks=16,
                     upsampling_depth=4,
                     enc_kernel_size=21,
                     enc_num_basis=512,
                    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # 训练工具：传入net的所有参数，设置学习率
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.5)
    optimizer = torch.optim.Adam(model.parameters(),lr=0.1,betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    # 损失函数
    loss_function = torch.nn.BCELoss(reduce = False,size_average=False) #损失函数
    # torch.nn.CrossEntropyLoss()
    # torch.nn.MSELoss()
    # torch.nn.BCELoss()
    best_testing_correct = 0
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99999)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=len(train_data),gamma=0.8)

    for epoch in range(epochs):
        print('Epoch:', epoch)
        train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=get_data_genertor,
                                      drop_last=True)
        for batch, (train, label) in enumerate(train_dataloader):
            train = torch.reshape(train,(batch_size,6,-1)).to(device)
            label_temp = y[label.numpy(),:]
            label = torch.from_numpy(label_temp).to(device)

            prediction = model(train.to(dtype=torch.float32)).to(device) # 输入x，输出预测值
            loss = loss_function(prediction, label)  # 计算预测值和真实值之间的误差
            loss.backward()  # 误差反向传播, 计算参数更新值
            optimizer.step()# 将参数更新值施加到model的 parameters 上
            scheduler.step()  # 指数衰减
            optimizer.zero_grad()  # 清空上一步的残余更新参数值
            print(loss)

        print("train ok ")

        testing_correct = 0
        valid_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=True, collate_fn=get_data_genertor,
                                      drop_last=True)
        for batch, (valid, label) in enumerate(valid_dataloader):
            valid = torch.reshape(train, (batch_size, 6, -1))
            label_temp = y[label.numpy(), :]
            label = torch.from_numpy(label_temp)

            prediction = model(train.to(dtype=torch.float32))  # 输入x，输出预测值
            testing_correct += torch.sum(abs(prediction-label))

        print("valid ok ")
        print("Test Accuracy is:{:.4f}".format(testing_correct))
        if epoch==0:
            best_testing_correct=testing_correct
        if testing_correct<best_testing_correct:
            best_testing_correct = testing_correct
            mpath = 'model_'+ str(epoch) + '.pkl'
            isExists = os.path.exists(mpath)
            if isExists:
                os.remove(mpath)
                    # os.makedirs(mpath)
            torch.save(model.state_dict(), mpath) #模型保存



