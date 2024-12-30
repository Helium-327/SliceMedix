# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2024/12/27 17:49:07
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: 训练和验证
=================================================
'''
import os

import time
import torch
import numpy as np
from tqdm import tqdm

from torch.amp import GradScaler, autocast

def train_one_epoch(model, train_loader, scaler, optimizer, loss_function, device):
    """
    ====训练过程====
    :param model: 模型
    :param metrics: 评估指标
    :param train_loader: 训练数据集
    :param val_loader: 验证数据集
    :param scaler: 缩放器
    :param optimizer: 优化器
    :param loss_funtion: 损失函数
    :param device: 设备
    :param model_path: 模型路径
    """
    model.train()
    
    train_running_loss = 0.0
    
    train_et_loss = 0.0
    train_tc_loss = 0.0
    train_wt_loss = 0.0
    
    train_loader = tqdm(train_loader, desc=f"🛠️--Training", leave=False)
    
    for data in train_loader: # 读取每一个 batch
        # 获取输入数据
        vimage, mask = data[0].to(device), data[1].to(device)
        
        # 梯度清零
        
        with autocast(device_type='cuda'): # 混合精度训练
            # 前向传播 + 反向传播 + 优化
            optimizer.zero_grad()
            predicted_mask = model(vimage)
            mean_loss, et_loss, tc_loss, wt_loss = loss_function(predicted_mask, mask)
        scaler.scale(mean_loss).backward()                           # 反向传播，只有训练模型时才需要
        scaler.step(optimizer)                                  # 优化器更新参数
        scaler.update()  
        
        train_running_loss += mean_loss.item()                       # 计算训练loss的累计和
        train_et_loss += et_loss.item() 
        train_tc_loss += tc_loss.item()
        train_wt_loss += wt_loss.item()
        
    return train_running_loss, train_et_loss, train_tc_loss, train_wt_loss

def val_one_epoch(model, Metric, val_loader, loss_function, epoch, device):
    """
    验证过程
    :param model: 模型
    :param metrics: 评估指标
    :param train_loader: 训练数据集
    :param val_loader: 验证数据集
    :param scaler: 缩放器
    :param optimizer: 优化器
    :param loss_funtion: 损失函数
    :param device: 设备
    :param model_path: 模型路径
    """
    val_running_loss = 0.0
    Metrics_list = np.zeros((7, 4))
    model.eval()
    val_et_loss = 0.0
    val_tc_loss = 0.0
    val_wt_loss = 0.0
    
    with torch.no_grad(): # 关闭梯度计算
        with autocast(device_type='cuda'):
            val_loader = tqdm(val_loader, desc=f"🧐--Validating", leave=False)
            for data in val_loader:
                vimage, mask = data[0].to(device), data[1].to(device)                
                with autocast(device_type='cuda'):
                    predicted_mask = model(vimage)
                    mean_loss, et_loss, tc_loss, wt_loss = loss_function(predicted_mask, mask)
                    metrics = Metric.update(predicted_mask, mask)
                    Metrics_list += metrics
                val_running_loss += mean_loss.item() 
                val_et_loss += et_loss.item() 
                val_tc_loss += tc_loss.item()
                val_wt_loss += wt_loss.item()
    
        
    return val_running_loss, val_et_loss, val_tc_loss, val_wt_loss, Metrics_list