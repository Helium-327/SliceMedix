# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2025/01/04 13:52:43
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: 基础卷积模块
*      VERSION: v1.0
=================================================
'''

import torch
from torch import nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
    
class ConvWithSE(nn.Module):
    def __init__(self, in_channels, out_channels, reduction_ratio=16):
        super(ConvWithSE, self).__init__()
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.se = SEBlock(out_channels, reduction_ratio)
        self.conv2 = ConvBlock(out_channels, out_channels)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.se(x)
        x = self.conv2(x)
        return x
    
class Down(nn.Module):
    def __init__(self, in_channels, out_channels, use_se=False):
        super(Down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvWithSE(in_channels, out_channels) if use_se else DoubleConv(in_channels, out_channels)
        )

if __name__ == '__main__':
    from Attention import SEBlock

else:
    from .Attention import SEBlock