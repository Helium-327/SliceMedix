# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2025/01/04 13:51:43
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: 深度可分离卷积模块
*      VERSION: v1.0
=================================================
'''
import torch
import torch.nn as nn

class DWConvBlcok(nn.Module):
    def __init__(
            self, in_channels,
            out_channels, 
            kernel_size=3, 
            padding=1, 
            use_residual=False, 
            activation=nn.ReLU
            ):
        super(DWConvBlcok, self).__init__()
        
        self.depthwise = nn.Conv2d(
            in_channels, 
            in_channels, 
            kernel_size=kernel_size, 
            padding=padding, 
            groups=in_channels
        )
        self.pointwise = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=1
        )
        
        self.activation = activation(inplace=True)
        self.use_residual = use_residual
        
        # 可选的残差连接
        if use_residual:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
            self.residual_activation = activation(inplace=True)
    
    def forward(self, x):
        identity = x
        
        x = self.depthwise(x)
        x = self.activation(x)
        
        x = self.pointwise(x)
        
        if self.use_residual:
            residual = self.residual(identity)
            x = x + residual
            x = self.residual_activation(x)
        
        return x
    
class DoubleDWConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleDWConv, self).__init__()
        self.conv1 = DWConvBlcok(in_channels, out_channels)
        self.conv2 = DWConvBlcok(out_channels, out_channels)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class DWConvWithSE(nn.Module):
    def __init__(self, in_channels, out_channels, reduction_ratio=16):
        super(DWConvWithSE, self).__init__()
        self.conv1 = DWConvBlcok(in_channels, out_channels)
        self.se = SEBlock(out_channels, reduction_ratio)
        self.conv2 = DWConvBlcok(out_channels, out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.se(x)
        x = self.conv2(x)
        return x
    

if __name__ == '__main__':
    from Attention import SEBlock

else:
    from .Attention import SEBlock

