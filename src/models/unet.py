# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2024/12/28 12:00:39
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: UNet模型

# !说明：
数据集： BRATS21 
输入：  (B, C, H, W) = (16, 144, 240, 240)
输出：  (B, C, H, W) = (16, 4, 240, 240)
=================================================
'''



import torch
from torch import nn

class EncoderBlcok(nn.Module):
    def __init__(self, in_channels, out_channels):
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        return x

class DecoderBlcok(nn.Module):
    def __init__(self, in_channels, out_channels):
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()

    def forward(self, x, skip=None):
        if skip is not None:
            x = torch.cat([skip, x], dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        return x




class EncoderAndDecoder(nn.Module):

    def __init__(self, in_channels, out_channels):
        """
        初始化 EncoderAndDecoder 类的实例。

        参数:
        - in_channels (int): 输入图像的通道数。
        - out_channels (int): 输出图像的通道数。
        """
        super(EncoderAndDecoder).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsampling = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.encoder1 = EncoderBlcok(in_channels, in_channels*2)
        self.encoder2 = EncoderBlcok(in_channels*2, in_channels*4)
        self.encoder3 = EncoderBlcok(in_channels*4, in_channels*8)
        self.encoder4 = EncoderBlcok(in_channels*8, in_channels*16)

        self.decoder1 = DecoderBlcok(in_channels*16, in_channels*8)
        self.decoder2 = DecoderBlcok(in_channels*8, in_channels*4)
        self.decoder3 = DecoderBlcok(in_channels*4, in_channels*2)
        self.decoder4 = DecoderBlcok(in_channels*2, in_channels)

    
    def forward(self, x):      # (16, 144, 240, 240)
        x1 = self.encoder1(x)  # 
        out = self.pooling(x1)

        x2 = self.encoder2(out)
        out = self.pooling(x2)

        x3 = self.encoder3(out)
        out = self.pooling(x3)

        x4 = self.encoder4(out)
        en_out = self.encoder4(out)

        x = self.decoder1(en_out)
        x = self.decoder2(x)
        x = self.decoder3(x)
        x = self.decoder4(x)

        return x
    

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        初始化 EncoderAndDecoder 类的实例。

        参数:
        - in_channels (int): 输入图像的通道数。
        - out_channels (int): 输出图像的通道数。
        """
        super(EncoderAndDecoder).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.encoder1 = EncoderBlcok(in_channels, in_channels*2)
        self.encoder2 = EncoderBlcok(in_channels*2, in_channels*4)
        self.encoder3 = EncoderBlcok(in_channels*4, in_channels*8)
        self.encoder4 = EncoderBlcok(in_channels*8, in_channels*16)

        self.decoder1 = DecoderBlcok(in_channels*16, in_channels*8)
        self.decoder2 = DecoderBlcok(in_channels*8, in_channels*4)
        self.decoder3 = DecoderBlcok(in_channels*4, in_channels*2)
        self.decoder4 = DecoderBlcok(in_channels*2, in_channels)

    
    def forward(self, x):
        x = self.encoder1(x)
        x = self.encoder2(x)
        x = self.encoder3(x)
        x = self.encoder4(x)

        x = self.decoder1(x)
        x = self.decoder2(x)
        x = self.decoder3(x)
        x = self.decoder4(x)

        return x
    
        


