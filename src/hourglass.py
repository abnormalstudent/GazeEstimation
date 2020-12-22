import torch
from torch import nn

def init_weights(layer):
    if isinstance(layer, nn.Conv2d):
        nn.init.kaiming_normal_(layer.weight)
        nn.init.zeros_(layer.bias)
    elif isinstance(layer, nn.Linear):
        nn.init.kaiming_normal_(layer.weight)
        nn.init.zeros_(layer.bias)
class Residual(nn.Module):
    def __init__(self, in_features, middle_features, use_batchnorm=False):
        super(Residual, self).__init__()
        self.use_batchnorm = use_batchnorm
        # Since we will always have to keep dimensions the same to produce heatmaps,
        # in_features == out_features
        if self.use_batchnorm:
            self.bn1 = nn.BatchNorm2d(middle_features)
            self.bn2 = nn.BatchNorm2d(middle_features)
            self.bn3 = nn.BatchNorm2d(in_features)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features, middle_features, kernel_size=[1, 1]),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(middle_features, middle_features, kernel_size=[3, 3], padding=[1, 1]),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(middle_features, in_features, kernel_size=[1, 1]),
            nn.ReLU()
        )
        
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        if self.use_batchnorm:
            out = self.bn1(out)
        
        out = self.conv2(out)
        if self.use_batchnorm:
            out = self.bn2(out)
        
        out = self.conv3(out)
        if self.use_batchnorm:
            out = self.bn3(out)
        
        out += residual
        return out

class Hourglass(nn.Module):
    def __init__(self, in_features=64, middle_features=32, residual_count = 3, use_batchnorm=False):
        self.residual_count = residual_count
        
        super(Hourglass, self).__init__()
        assert self.residual_count >= 1, "Residual count must be >= 1"
        # We are in the center of the network,
        # hence we need just one residual layer
        if self.residual_count == 1:
            self.inner_process = nn.Sequential()
            for i in range(3):
                self.inner_process.add_module(
                    "inner_residual_{}".format(i),
                    Residual(in_features, middle_features, use_batchnorm = use_batchnorm)
                )
            return 
        # We are outside of center, hence we need to recursively define out network
        
        # First downsample and apply residual
        self.down_sample = nn.Sequential(
            nn.MaxPool2d(kernel_size=[2, 2], stride = [2, 2]),
            Residual(in_features, middle_features, use_batchnorm=use_batchnorm)
        )
        # Magic of recursion
        self.inner_hourglass = Hourglass(
            in_features=in_features,
            middle_features=middle_features,
            residual_count = self.residual_count - 1,
            use_batchnorm=use_batchnorm
        )
        # Upsample
        self.up_sample = nn.Sequential(
            # Should test different interpolation methods
            nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True),
            Residual(in_features, middle_features, use_batchnorm=use_batchnorm)
        )    
        # Don't forget about transformation
        self.transform = Residual(in_features, middle_features, use_batchnorm=use_batchnorm)
        
    def forward(self, x):
        if self.residual_count == 1:
            return self.inner_process(x)
        else:
            x = self.down_sample(x)
            
            inner_hourglass_out = self.inner_hourglass(x)
            transformation_out = self.transform(x)
            
            x = self.up_sample(transformation_out + inner_hourglass_out)
            return x