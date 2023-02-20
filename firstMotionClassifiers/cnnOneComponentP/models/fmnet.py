#!/usr/bin/env python3
import torch
import h5py
import numpy as np

class FMNet(torch.nn.Module):

    def __init__(self, num_channels = 1, num_classes = 3):
        super(FMNet, self).__init__()
        from torch.nn import MaxPool1d, Conv1d, Linear
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim = 1)
        filter1 = 21
        filter2 = 15
        filter3 = 11

        self.maxpool = MaxPool1d(kernel_size=2, stride=2)
        self.conv1 = Conv1d(num_channels, 32,
                            kernel_size=filter1, padding=filter1//2)
        self.bn1 = torch.nn.BatchNorm1d(32, eps=1e-05, momentum=0.1)
        # Output has dimension [200 x 32]


        self.conv2 = Conv1d(32, 64,
                            kernel_size=filter2, padding=filter2//2)
        self.bn2 = torch.nn.BatchNorm1d(64, eps=1e-05, momentum=0.1)
        # Output has dimension [100 x 64] 

        self.conv3 = Conv1d(64, 128,
                            kernel_size=filter3, padding=filter3//2)
        self.bn3 = torch.nn.BatchNorm1d(128, eps=1e-05, momentum=0.1)
        # Output has dimension [50 x 128]

        self.fcn1 = Linear(6400, 512)
        self.bn4 = torch.nn.BatchNorm1d(512, eps=1e-05, momentum=0.1)

        self.fcn2 = Linear(512, 512)
        self.bn5 = torch.nn.BatchNorm1d(512, eps=1e-05, momentum=0.1)

        self.fcn3 = Linear(512, 3)

    def forward(self, x): 
        # N.B. Consensus seems to be growing that BN goes after nonlinearity
        # That's why this is different than Zach's original paper.
        # First convolutional layer
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.maxpool(x)
        # Second convolutional layer
        x = self.conv2(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.maxpool(x)
        # Third convolutional layer
        x = self.conv3(x)
        x = self.relu(x)
        x = self.bn3(x)
        x = self.maxpool(x)
        # Flatten
        x = x.flatten(1) #torch.nn.flatten(x)
        # First fully connected layer
        x = self.fcn1(x)
        x = self.relu(x)
        x = self.bn4(x)
        # Second fully connected layer
        x = self.fcn2(x)
        x = self.relu(x)
        x = self.bn5(x)
        # Last layer
        x = self.fcn3(x)
        # Squash outputs to [0,1]
        x = self.softmax(x)
        return x

    def write_weights_to_hdf5(self, file_name):
        f = h5py.File(file_name, 'w')
        g = f.create_group("/model_weights")

        g.create_dataset("conv1d_1.weight", data=np.array(self.conv1.weight.data.cpu()))
        g.create_dataset("conv1d_1.bias", data=np.array(self.conv1.bias.data.cpu()))
        g.create_dataset("bn_1.weight", data=np.array(self.bn1.weight.data.cpu())) # gamma
        g.create_dataset("bn_1.bias", data=np.array(self.bn1.bias.data.cpu()))  # beta
        g.create_dataset("bn_1.running_mean", data=np.array(self.bn1.running_mean.data.cpu()))
        g.create_dataset("bn_1.running_var", data=np.array(self.bn1.running_var.data.cpu()))

        g.create_dataset("conv1d_2.weight", data=np.array(self.conv2.weight.data.cpu()))
        g.create_dataset("conv1d_2.bias", data=np.array(self.conv2.bias.data.cpu()))
        g.create_dataset("bn_2.weight", data=np.array(self.bn2.weight.data.cpu())) # gamma
        g.create_dataset("bn_2.bias", data=np.array(self.bn2.bias.data.cpu()))  # beta
        g.create_dataset("bn_2.running_mean", data=np.array(self.bn2.running_mean.data.cpu()))
        g.create_dataset("bn_2.running_var", data=np.array(self.bn2.running_var.data.cpu()))

        g.create_dataset("conv1d_3.weight", data=np.array(self.conv3.weight.data.cpu()))
        g.create_dataset("conv1d_3.bias", data=np.array(self.conv3.bias.data.cpu()))
        g.create_dataset("bn_3.weight", data=np.array(self.bn3.weight.data.cpu())) # gamma
        g.create_dataset("bn_3.bias", data=np.array(self.bn3.bias.data.cpu()))  # beta
        g.create_dataset("bn_3.running_mean", data=np.array(self.bn3.running_mean.data.cpu()))
        g.create_dataset("bn_3.running_var", data=np.array(self.bn3.running_var.data.cpu()))

        g.create_dataset("fcn_1.weight", data=np.array(self.fcn1.weight.data.cpu()))
        g.create_dataset("fcn_1.bias", data=np.array(self.fcn1.bias.data.cpu()))
        g.create_dataset("bn_4.weight", data=np.array(self.bn4.weight.data.cpu())) # gamma
        g.create_dataset("bn_4.bias", data=np.array(self.bn4.bias.data.cpu()))  # beta
        g.create_dataset("bn_4.running_mean", data=np.array(self.bn4.running_mean.data.cpu()))
        g.create_dataset("bn_4.running_var", data=np.array(self.bn4.running_var.data.cpu()))

        g.create_dataset("fcn_2.weight", data=np.array(self.fcn2.weight.data.cpu()))
        g.create_dataset("fcn_2.bias", data=np.array(self.fcn2.bias.data.cpu()))
        g.create_dataset("bn_5.weight", data=np.array(self.bn5.weight.data.cpu())) # gamma
        g.create_dataset("bn_5.bias", data=np.array(self.bn5.bias.data.cpu()))  # beta
        g.create_dataset("bn_5.running_mean", data=np.array(self.bn5.running_mean.data.cpu()))
        g.create_dataset("bn_5.running_var", data=np.array(self.bn5.running_var.data.cpu()))

        g.create_dataset("fcn_3.weight", data=np.array(self.fcn3.weight.data.cpu()))
        g.create_dataset("fcn_3.bias", data=np.array(self.fcn3.bias.data.cpu()))

        f.close()

