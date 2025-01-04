# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2025/01/04 13:39:36
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: 各种注意力模块
*      VERSION: v1.0
=================================================
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        """
        使用1x1卷积的SE注意力模块
        
        参数:
        - in_channels: 输入通道数
        - reduction_ratio: 通道压缩比例
        """
        super(SEBlock, self).__init__()
        
        # 确保压缩后的通道数至少为1
        reduced_channels = max(1, in_channels // reduction_ratio)
        
        self.se = nn.Sequential(
            # 空间信息压缩
            nn.AdaptiveAvgPool2d(1),
            
            # 第一个1x1卷积层 - 降维
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(inplace=True),
            
            # 第二个1x1卷积层 - 升维
            nn.Conv2d(reduced_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # 生成通道注意力权重
        attention = self.se(x)

        # 应用注意力权重
        return x * attention
    
class EMA(nn.Module):
    def __init__(self, channels, c2=None, factor=32):
        super(EMA, self).__init__()
        self.group = factor
        assert channels // self.group > 0
        self.softmax = nn.Softmax(dim=-1)
        self.averagePooling = nn.AdaptiveAvgPool2d((1,1))
        self.Pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.Pool_w = nn.AdaptiveAvgPool2d((1, None))

        self.groupNorm = nn.GroupNorm(channels // self.group, channels//self.group)
        self.conv1x1 = nn.Conv2d(channels // self.group, channels // self.group, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.group, channels // self.group, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b*self.group, -1, h, w)
        x_h = self.Pool_h(group_x)  # 高度方向池化
        x_w = self.Pool_w(group_x).permute(0, 1, 3, 2)  # 宽度方向池化

        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2)) # 拼接之后卷积
        x_h, x_w = torch.split(hw, [h, w], dim=2)       # 拆分

        # 1x1路径
        x1 = self.groupNorm(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())          # 高度的注意力
        x11 = self.softmax(self.averagePooling(x1).reshape(b*self.group, -1, 1).permute(0, 2, 1)) # 对 x1 进行平均池化，然后进行 softmax 操作
        x12 = x1.reshape(b*self.group, c//self.group, -1)

        # 3x3路径
        x2 = self.conv3x3(group_x) # 通过 3x3卷积层
        x21 = self.softmax(self.averagePooling(x2).reshape(b*self.group, -1, 1).permute(0, 2, 1)) # 对 x2 进行平均池化，然后进行 softmax 操作
        x22 = x2.reshape(b*self.group, c//self.group, -1)

        weights = (torch.matmul(x11, x22) + torch.matmul(x21, x12)).reshape(b * self.group, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)

class EMA_pro(nn.Module):
    """改进的EMA模块"""
    def __init__(self, channels, factor=32, use_residual=True, act_layer=nn.SiLU):
        """
        Enhanced Memory Attention Module
        
        参数:
        - channels: 输入通道数
        - factor: 分组因子，用于控制特征分组数量
        - use_residual: 是否使用残差连接
        - act_layer: 激活函数类型
        """
        super(EMA_pro, self).__init__()
        
        self.groups = factor
        self.use_residual = use_residual
        
        # 确保通道数可以被factor整除
        assert channels % self.groups == 0, f"Channels ({channels}) must be divisible by factor ({factor})"
        self.group_channels = channels // self.groups
        
        # 空间池化层
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # H×1
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # 1×W
        self.pool_hw = nn.AdaptiveAvgPool2d((1, 1))    # 1×1
        
        # 特征变换层
        self.gn = nn.GroupNorm(self.group_channels, self.group_channels)
        self.conv_reduce = nn.Sequential(
            nn.Conv2d(self.group_channels, self.group_channels, 1, bias=False),
            nn.BatchNorm2d(self.group_channels),
            act_layer()
        )
        self.conv_spatial = nn.Sequential(
            nn.Conv2d(self.group_channels, self.group_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.group_channels),
            act_layer()
        )
        
        # 初始化
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _split_groups(self, x):
        b, c, h, w = x.shape
        return x.reshape(b * self.groups, -1, h, w)
    
    def _merge_groups(self, x, b):
        _, c, h, w = x.shape
        return x.reshape(b, -1, h, w)
    
    def _compute_attention_weights(self, group_x, x_h, x_w):
        """计算注意力权重"""
        b, c, h, w = group_x.shape
        
        # 计算方向注意力
        spatial_attn = group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid()
        spatial_attn = self.gn(spatial_attn)
        
        # 计算通道注意力
        channel_attn1 = self._compute_channel_attention(spatial_attn)
        channel_attn2 = self._compute_channel_attention(self.conv_spatial(group_x))
        
        # 融合注意力
        attn = channel_attn1 + channel_attn2
        return attn.sigmoid()
    
    def _compute_channel_attention(self, x):
        b, c, h, w = x.shape
        # 计算通道注意力权重
        x_pool = self.pool_hw(x).reshape(b, -1, 1)
        x_pool = F.softmax(x_pool, dim=1).permute(0, 2, 1)
        x_feat = x.reshape(b, c, -1)
        
        return torch.matmul(x_pool, x_feat).reshape(b, 1, h, w)
    
    def forward(self, x):
        identity = x
        b, _, h, w = x.shape
        
        # 分组处理
        group_x = self._split_groups(x)
        
        # 计算空间注意力
        x_h = self.conv_reduce(self.pool_h(group_x))
        x_w = self.conv_reduce(self.pool_w(group_x))
        
        # 计算注意力权重并应用
        attn_weights = self._compute_attention_weights(group_x, x_h, x_w)
        out = group_x * attn_weights
        
        # 合并分组
        out = self._merge_groups(out, b)
        
        # 残差连接
        if self.use_residual:
            out = out + identity
            
        return out


class SelfAttention3D(nn.Module):
    """自注意力模块"""
    def __init__(self, in_channels):
        super(SelfAttention3D, self).__init__()
        self.query_conv =   nn.Conv3d(in_channels, in_channels // 8, kernel_size=1) # 全连接
        self.key_conv   =   nn.Conv3d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv =   nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, D, H, W = x.size()
        
        # Permute and reshape the input to separate batch and spatial dimensions
        query = self.query_conv(x).view(batch_size, -1, D, H, W).permute(0, 2, 3, 4, 1).contiguous()
        key = self.key_conv(x).view(batch_size, -1, D, H, W).permute(0, 2, 3, 4, 1).contiguous()
        value = self.value_conv(x).view(batch_size, -1, D, H, W).permute(0, 2, 3, 4, 1).contiguous()
        
        # Calculate the attention scores
        energy = torch.matmul(query, key.transpose(-1, -2))
        attention = F.softmax(energy, dim=-1)
        
        # Apply the attention to the values
        out = torch.matmul(attention, value)
        
        # Reshape and permute the output to match the original input shape
        out = out.permute(0, 4, 1, 2, 3).contiguous().view(batch_size, C, D, H, W)
        
        # Scale and add a residual connection
        out = self.gamma * out + x
        
        return out