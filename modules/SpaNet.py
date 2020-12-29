import torch
import torch.nn as nn

from modules.Hourglass import Hourglass
from modules.DenseNet import DenseNet

class SpaNet(nn.Module):
    def __init__(self, in_features=64, middle_features=32, residual_count=3, use_batchnorm=True):
        super(SpaNet, self).__init__()
        # (N, 3, 80, 120)
        self.pre_conv = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Conv2d(in_channels=3, out_channels=in_features, kernel_size=[3, 3], padding=[1, 1])
        )
        # (N, 64, 80, 120)
        self.hourglass1= Hourglass(in_features=in_features, middle_features=middle_features, \
                                   residual_count=residual_count, use_batchnorm=use_batchnorm)
        self.hourglass2= Hourglass(in_features=in_features, middle_features=middle_features, \
                                   residual_count=residual_count, use_batchnorm=use_batchnorm)
        self.hourglass3= Hourglass(in_features=in_features, middle_features=middle_features, \
                                   residual_count=residual_count, use_batchnorm=use_batchnorm)
        self.dense_net = DenseNet(growth_rate=8, \
                                  compression_factor=0.5, \
                                  num_layers_per_block=(4, 4, 4, 4), \
                                  dense_net_input_features=in_features)
    def forward(self, x):
        x = self.pre_conv(x)
        x = self.hourglass1(x)
        x = self.hourglass2(x)
        x = self.hourglass3(x)
        return self.dense_net(x)
