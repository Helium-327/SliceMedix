# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2024/12/28 12:00:39
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: UNet模型

# !说明：
数据集： BRATS21 
输入：  (B, C, H, W) = (16, 4, 240, 240)
输出：  (B, C, H, W) = (16, 4, 240, 240)
=================================================
'''

import torch
from torch import nn
from torchinfo import summary
# from src.nnArchitecture.modules.ConvBlock import *
    
class UNet(nn.Module):
    def __init__(self, in_channels=4, mid_channels=32, out_channels=4):
        super(UNet, self).__init__()
        """
        初始化 EncoderAndDecoder 类的实例。

        参数:
        - in_channels (int): 输入图像的通道数。
        - out_channels (int): 输出图像的通道数。
        """
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsampling = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.encoder1 = DoubleConv(in_channels, mid_channels*2)           # 4 -> 64
        self.encoder2 = DoubleConv(mid_channels*2, mid_channels*4)        # 64 -> 128
        self.encoder3 = DoubleConv(mid_channels*4, mid_channels*8)        # 128 -> 256
        self.encoder4 = DoubleConv(mid_channels*8, mid_channels*16)       # 256 -> 512

        self.decoder1 = DoubleConv(mid_channels*32, mid_channels*8)       # 1024 -> 256
        self.decoder2 = DoubleConv(mid_channels*16, mid_channels*4)        # 512 -> 128
        self.decoder3 = DoubleConv(mid_channels*8, mid_channels*2)        # 256 -> 64
        self.decoder4 = DoubleConv(mid_channels*4, mid_channels)          # 128 -> 32


        self.out_conv = nn.Conv2d(mid_channels, out_channels, kernel_size=1, padding=0)

        self.softmax = nn.Softmax(dim=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):    # (16, 4, 240, 240)

        # 编码器
        skip_1 = self.encoder1(x)           #  (16, 64, 240, 240)
        x = self.pooling(skip_1)          #  (16, 64, 120, 120)

        skip_2 = self.encoder2(x)         #  (16, 128, 120, 120)   
        x = self.pooling(skip_2)          #  (16, 128, 60, 60)

        skip_3 = self.encoder3(x)         #  (16, 256, 60, 60)    
        x = self.pooling(skip_3)          #  (16, 256, 30, 30)

        skip_4 = self.encoder4(x)         #  (16, 512, 30, 30)    
        en_out = self.pooling(skip_4)     #  (16, 512, 15, 15)

        # 中间层

        # 解码器
        x = torch.cat([self.upsampling(en_out), skip_4], dim=1)  # (16, 1024, 30, 30)
        x = self.decoder1(x)                                     # (16, 256, 30, 30)

        x = torch.cat([self.upsampling(x), skip_3], dim=1)       # (16, 512, 60, 60)
        x = self.decoder2(x)                                     # (16, 128, 60, 60)

        x = torch.cat([self.upsampling(x), skip_2], dim=1)       # (16, 256, 120, 120)
        x = self.decoder3(x)                                     # (16, 64, 120, 120)    

        x = torch.cat([self.upsampling(x), skip_1], dim=1)       # (16, 128, 240, 240)
        x = self.decoder4(x)                                     # (16, 32, 240, 240)

        out = self.out_conv(x)                                   # (16, 4, 240, 240)

        out = self.softmax(out)

        return out

class DWConvUNet(nn.Module):
    def __init__(self, in_channels=4, mid_channels=32, out_channels=4):
        super(DWConvUNet, self).__init__()
        """
        初始化 EncoderAndDecoder 类的实例。

        参数:
        - in_channels (int): 输入图像的通道数。
        - out_channels (int): 输出图像的通道数。
        """
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsampling = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.encoder1 = DoubleDWConv(in_channels, mid_channels*2)           # 4 -> 64
        self.encoder2 = DoubleDWConv(mid_channels*2, mid_channels*4)        # 64 -> 128
        self.encoder3 = DoubleDWConv(mid_channels*4, mid_channels*8)        # 128 -> 256
        self.encoder4 = DoubleDWConv(mid_channels*8, mid_channels*16)       # 256 -> 512

        self.decoder1 = DoubleDWConv(mid_channels*32, mid_channels*8)       # 1024 -> 256
        self.decoder2 = DoubleDWConv(mid_channels*16, mid_channels*4)        # 512 -> 128
        self.decoder3 = DoubleDWConv(mid_channels*8, mid_channels*2)        # 256 -> 64
        self.decoder4 = DoubleDWConv(mid_channels*4, mid_channels)          # 128 -> 32


        self.out_conv = nn.Conv2d(mid_channels, out_channels, kernel_size=1, padding=0)

        self.softmax = nn.Softmax(dim=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)    

    def forward(self, x):    # (16, 4, 240, 240)

        # 编码器
        skip_1 = self.encoder1(x)           #  (16, 64, 240, 240)
        x = self.pooling(skip_1)          #  (16, 64, 120, 120)

        skip_2 = self.encoder2(x)         #  (16, 128, 120, 120)   
        x = self.pooling(skip_2)          #  (16, 128, 60, 60)

        skip_3 = self.encoder3(x)         #  (16, 256, 60, 60)    
        x = self.pooling(skip_3)          #  (16, 256, 30, 30)

        skip_4 = self.encoder4(x)         #  (16, 512, 30, 30)    
        en_out = self.pooling(skip_4)     #  (16, 512, 15, 15)

        # 中间层

        # 解码器
        x = torch.cat([self.upsampling(en_out), skip_4], dim=1)  # (16, 1024, 30, 30)
        x = self.decoder1(x)                                     # (16, 256, 30, 30)

        x = torch.cat([self.upsampling(x), skip_3], dim=1)       # (16, 512, 60, 60)
        x = self.decoder2(x)                                     # (16, 128, 60, 60)

        x = torch.cat([self.upsampling(x), skip_2], dim=1)       # (16, 256, 120, 120)
        x = self.decoder3(x)                                     # (16, 64, 120, 120)    

        x = torch.cat([self.upsampling(x), skip_1], dim=1)       # (16, 128, 240, 240)
        x = self.decoder4(x)                                     # (16, 32, 240, 240)

        out = self.out_conv(x)                                   # (16, 4, 240, 240)

        out = self.softmax(out)

        return out
    

class DWConvUNet(nn.Module):
    def __init__(self, in_channels=4, mid_channels=32, out_channels=4):
        super(DWConvUNet, self).__init__()
        """
        初始化 EncoderAndDecoder 类的实例。

        参数:
        - in_channels (int): 输入图像的通道数。
        - out_channels (int): 输出图像的通道数。
        """
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        # 使用转置卷积代替 nn.Upsample
        self.upsample1 = nn.ConvTranspose2d(mid_channels * 16, mid_channels * 8, kernel_size=2, stride=2)
        self.upsample2 = nn.ConvTranspose2d(mid_channels * 8, mid_channels * 4, kernel_size=2, stride=2)
        self.upsample3 = nn.ConvTranspose2d(mid_channels * 4, mid_channels * 2, kernel_size=2, stride=2)
        self.upsample4 = nn.ConvTranspose2d(mid_channels * 2, mid_channels, kernel_size=2, stride=2)

        self.encoder1 = DoubleConv(in_channels, mid_channels*2)           # 4 -> 64
        self.encoder2 = DoubleConv(mid_channels*2, mid_channels*4)        # 64 -> 128
        self.encoder3 = DoubleConv(mid_channels*4, mid_channels*8)        # 128 -> 256
        self.encoder4 = DoubleConv(mid_channels*8, mid_channels*16)       # 256 -> 512

        self.decoder1 = DoubleConv(mid_channels*32, mid_channels*8)       # 1024 -> 256
        self.decoder2 = DoubleConv(mid_channels*16, mid_channels*4)        # 512 -> 128
        self.decoder3 = DoubleConv(mid_channels*8, mid_channels*2)        # 256 -> 64
        self.decoder4 = DoubleConv(mid_channels*4, mid_channels)          # 128 -> 32


        self.out_conv = nn.Conv2d(mid_channels, out_channels, kernel_size=1, padding=0)

        self.softmax = nn.Softmax(dim=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):    # (16, 4, 240, 240)

        if torch.isnan(x).any() or torch.isinf(x).any():
            raise ValueError("Input tensor contains NaN or Inf values.")
        # 编码器
        skip_1 = self.encoder1(x)           #  (16, 64, 240, 240)
        x = self.pooling(skip_1)          #  (16, 64, 120, 120)

        skip_2 = self.encoder2(x)         #  (16, 128, 120, 120)   
        x = self.pooling(skip_2)          #  (16, 128, 60, 60)

        skip_3 = self.encoder3(x)         #  (16, 256, 60, 60)    
        x = self.pooling(skip_3)          #  (16, 256, 30, 30)

        skip_4 = self.encoder4(x)         #  (16, 512, 30, 30)    
        en_out = self.pooling(skip_4)     #  (16, 512, 15, 15)

        # 中间层

        # 解码器
        x = torch.cat([self.upsample1(en_out), skip_4], dim=1)  # (16, 1024, 30, 30)
        x = self.decoder1(x)                                     # (16, 256, 30, 30)

        x = torch.cat([self.upsample2(x), skip_3], dim=1)       # (16, 512, 60, 60)
        x = self.decoder2(x)                                     # (16, 128, 60, 60)

        x = torch.cat([self.upsample3(x), skip_2], dim=1)       # (16, 256, 120, 120)
        x = self.decoder3(x)                                     # (16, 64, 120, 120)    

        x = torch.cat([self.upsample4(x), skip_1], dim=1)       # (16, 128, 240, 240)
        x = self.decoder4(x)                                     # (16, 32, 240, 240)

        out = self.out_conv(x)                                   # (16, 4, 240, 240)

        out = self.softmax(out)
        if torch.isnan(out).any() or torch.isinf(out).any():
            raise ValueError("Output tensor contains NaN or Inf values.")

        return out   

if __name__ == '__main__':
    from modules.BasicConvBlock import *

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 原始UNet
    print("Depthwise UNet Summary:")
    model_original = UNet(4, 32, 4).to(device)
    summary(model_original, input_size=(16, 4, 240, 240), 
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"])
    
    print("\n" + "="*50 + "\n")
    
    # 深度可分离卷积UNet
    print("Depthwise Separable UNet Summary:")

    # 计算参数量的减少百分比
    params_original = sum(p.numel() for p in model_original.parameters())
    # params_depthwise = sum(p.numel() for p in model_depthwise.parameters())
    # reduction = (params_original - params_depthwise) / params_original * 100
    
    # print(f"\nParameter Reduction: {reduction:.2f}%")
    print(f"Original Parameters: {params_original:,}")
    # print(f"Depthwise Parameters: {params_depthwise:,}")
else:
    from nnArchitecture.modules.BasicConvBlock import *
    from nnArchitecture.modules.DWConvsBlcok import *
    




