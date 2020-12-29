import torch
import torch.nn as nn

class Composite(nn.Module):
    def __init__(self, in_channels=3, out_channels=8, kernel_size=[3, 3], padding=[1, 1]):
        super(Composite, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                              kernel_size=kernel_size, padding=padding)
    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        return x
class DenseBlock(nn.Module):
    def __init__(self, number_of_layers=4, in_channels=32, growth_rate=8):
        super(DenseBlock, self).__init__()
        self.number_of_layers = number_of_layers
        self.composites = nn.Sequential()
        for i in range(number_of_layers):
            self.composites.add_module(
                "composite_{}".format(i),
                Composite(in_channels + growth_rate * i, growth_rate)
            )
    def forward(self, x):
        for i in range(len(self.composites)):
            x_prev = x
            x = self.composites[i](x)
            x = torch.cat([x, x_prev], dim=1)
        return x
class DenseNet(nn.Module):
    def __init__(self, growth_rate=8, 
                 compression_factor=0.5, 
                 num_layers_per_block=(4, 4, 4, 4), 
                 dense_net_input_features=64):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate
        self.compression_factor = compression_factor
        self.num_layers_per_block = num_layers_per_block
        self.number_of_dense_blocks = len(num_layers_per_block)
        
        self.pre_conv = nn.Conv2d(in_channels=dense_net_input_features, \
                                  out_channels=2 * self.growth_rate, kernel_size=[1, 1])
        
        self.dense_blocks = nn.Sequential()
        self.transition_layers = nn.Sequential()
        
        initial_feature_maps = 2 * self.growth_rate
        dense_block_out = None  
        transition_out = None
        for i in range(self.number_of_dense_blocks):
            real_input = initial_feature_maps if transition_out is None else transition_out
            self.dense_blocks.add_module(
                "dense_block_{}".format(i),
                DenseBlock(self.num_layers_per_block[i], in_channels=real_input, growth_rate=self.growth_rate)
            )
            # Since we did torch.cat "self.num_l_p_b[i]" time, it growed like that
            dense_block_out = self.num_layers_per_block[i] * self.growth_rate + real_input
            
            # Use kind of combination of both transition and bottleneck layers
            transition_out = int(self.compression_factor * min(4 * self.growth_rate, dense_block_out))
            if i + 1 != self.number_of_dense_blocks:
                self.transition_layers.add_module(
                    "transition_layer_{}".format(i),
                    nn.Sequential(
                        Composite(dense_block_out, transition_out, kernel_size=[1, 1], padding=[0, 0]),
                        nn.AvgPool2d(kernel_size=[2, 2], stride=2)
                    )
                ) 
        self.flatten = nn.Flatten()
        self.regressor = nn.Linear(dense_block_out, 2)
        
    def forward(self, x):
        x = self.pre_conv(x)
        # x.shape = (N, 2 * self.growth_rate, H, W)
        for i in range(self.number_of_dense_blocks):
            x = self.dense_blocks[i](x)
            if i + 1 != self.number_of_dense_blocks:
                x = self.transition_layers[i](x)
        x = torch.mean(x, dim = (2, 3))
        x = self.flatten(x)
        return self.regressor(x)