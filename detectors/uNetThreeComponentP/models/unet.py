#!/usr/bin/env python3
import torch

class UNet(torch.nn.Module):

    def __init__(self, num_channels=3, num_classes=1, k=3):
        super(UNet, self).__init__()
        from torch.nn import MaxPool1d, Conv1d, ConvTranspose1d
        self.relu = torch.nn.ReLU()
        #k = 3 #7
        p = k//2
        self.maxpool = MaxPool1d(kernel_size=2, stride=2)
        self.conv11 = Conv1d(num_channels, 64, kernel_size=k, padding=p)
        self.conv12 = Conv1d(64, 64, kernel_size=k, padding=p)
        self.bn1 = torch.nn.BatchNorm1d(64, eps=1e-05, momentum=0.1)

        self.conv21 = Conv1d(64, 128, kernel_size=k, padding=p)
        self.conv22 = Conv1d(128, 128, kernel_size=k, padding=p)
        self.bn2 = torch.nn.BatchNorm1d(128, eps=1e-05, momentum=0.1)

        self.conv31 = Conv1d(128, 256, kernel_size=k, padding=p)
        self.conv32 = Conv1d(256, 256, kernel_size=k, padding=p)
        self.bn3 = torch.nn.BatchNorm1d(256, eps=1e-05, momentum=0.1)

        self.conv41 = Conv1d(256, 512, kernel_size=k, padding=p)
        self.conv42 = Conv1d(512, 512, kernel_size=k, padding=p)
        self.bn4 = torch.nn.BatchNorm1d(512, eps=1e-05, momentum=0.1)

        self.conv51 = Conv1d(512, 1024, kernel_size=k, padding=p)
        self.conv52 = Conv1d(1024, 1024, kernel_size=k, padding=p)
        self.bn5 = torch.nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1)

        self.uconv6 = ConvTranspose1d(1024, 512, kernel_size=2, stride=2)
        self.conv61 = Conv1d(1024, 512, kernel_size=k, padding=p)
        self.conv62 = Conv1d(512, 512, kernel_size=k, padding=p)
        self.bn6 = torch.nn.BatchNorm1d(512, eps=1e-05, momentum=0.1)

        self.uconv7 = ConvTranspose1d(512, 256, kernel_size=2, stride=2)
        self.conv71 = Conv1d(512, 256, kernel_size=k, padding=p)
        self.conv72 = Conv1d(256, 256, kernel_size=k, padding=p)
        self.bn7 = torch.nn.BatchNorm1d(256, eps=1e-05, momentum=0.1)

        self.uconv8 = ConvTranspose1d(256, 128, kernel_size=2, stride=2)
        self.conv81 = Conv1d(256, 128, kernel_size=k, padding=p)
        self.conv82 = Conv1d(128, 128, kernel_size=k, padding=p)
        self.bn8 = torch.nn.BatchNorm1d(128, eps=1e-05, momentum=0.1)

        self.uconv9 = ConvTranspose1d(128, 64, kernel_size=2, stride=2)
        self.conv91 = Conv1d(128, 64, kernel_size=k, padding=p)
        self.conv92 = Conv1d(64, 64, kernel_size=k, padding=p)
        self.bn9 = torch.nn.BatchNorm1d(64, eps=1e-05, momentum=0.1)

        self.conv93 = Conv1d(64, num_classes, kernel_size=1, padding=0)

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.conv11(x)
        x = self.relu(x)
        x = self.conv12(x)
        x1d = self.relu(x)
        x1d = self.bn1(x1d)
        x = self.maxpool(x1d)

        x = self.conv21(x)
        x = self.relu(x)
        x = self.conv22(x)
        x2d = self.relu(x)
        x2d = self.bn2(x2d)
        x = self.maxpool(x2d)
        #print('x2d.shape', x2d.shape)

        x = self.conv31(x)
        x = self.relu(x)
        x = self.conv32(x)
        x3d = self.relu(x)
        x3d = self.bn3(x3d)
        x = self.maxpool(x3d)
        #print('x3d.shape:', x3d.shape)

        x = self.conv41(x)
        x = self.relu(x)
        x = self.conv42(x)
        x4d = self.relu(x)
        x4d = self.bn4(x4d)
        x = self.maxpool(x4d)
        #print('x4d.shape:', x4d.shape)

        x = self.conv51(x)
        x = self.relu(x)
        x = self.conv52(x)
        x5d = self.relu(x)
        x5d = self.bn5(x5d)
        #print(x5d.shape)

        x6u = self.uconv6(x5d)
        #print(x6u.shape, x4d.shape)
        x = torch.cat((x4d, x6u), 1)
        x = self.conv61(x)
        x = self.relu(x)
        x = self.conv62(x)
        x = self.relu(x)
        x = self.bn6(x)

        x7u = self.uconv7(x)
        x = torch.cat((x3d, x7u), 1)
        x = self.conv71(x)
        x = self.relu(x)
        x = self.conv72(x)
        x = self.relu(x)
        x = self.bn7(x)
        #print('x.shape at 7', x.shape)

        x8u = self.uconv8(x)
        x = torch.cat((x2d, x8u), 1)
        x = self.conv81(x)
        x = self.relu(x)
        x = self.conv82(x)
        x = self.relu(x)
        x = self.bn8(x)

        #print
        x9u = self.uconv9(x)
        x = torch.cat((x1d, x9u), 1)
        x = self.conv91(x)
        x = self.relu(x)
        x = self.conv92(x)
        x = self.relu(x)
        x = self.bn9(x)

        x = self.conv93(x)

        # Sigmoid is 1/(1 + exp(-x))
        # Float max: float: 3.40282e+38
        # log(float max): 88.7
        x = torch.clamp(x, min = -87, max = 87)
        x = self.sigmoid(x)

        return x

