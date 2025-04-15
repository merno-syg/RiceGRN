import torch
import torch.nn as nn
from torch.nn import functional as F



class ConvBlock(nn.Module):
    def __init__(self, in_feature, hidden_channels, strides=1, residual=False, use_1x1conv=False):
        super().__init__()

        self.conv1 = nn.Conv2d(in_feature, hidden_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_feature, hidden_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        if self.conv3:
            x = self.conv3(x)
        x += residual
        return F.relu(x)


class CNN1dNet(nn.Module):
    def __init__(self, in_feature, num_layers, hidden_channels, classes, residual=False, first_block=True):
        super().__init__()
        blocks = []
        for _ in range(num_layers):
            if _ == 0 and first_block:
                blocks.append(ConvBlock(in_feature, hidden_channels, residual=residual, use_1x1conv=True))
            else:
                blocks.append(ConvBlock(in_feature, hidden_channels, residual=residual, use_1x1conv=False))
        return blocks



