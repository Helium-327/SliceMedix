# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2024/12/30 14:07:28
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: 训练和验证
=================================================
'''

from tqdm import tqdm
from torch.amp import GradScaler, autocast
import torch
import numpy as np


def train_one_epoch(model, train_loader, optimizer, loss_func, scaler, device):
    """
    训练一个epoch
    :param model: 模型
    :param train_loader: 训练数据加载器
    :param optimizer: 优化器
    :param loss_func: 损失函数
    :param scaler: 梯度缩放器
    :param device: 设备
    :return: 平均损失值
    """
    model.train()

    total_loss, total_et_loss, total_tc_loss, total_wt_loss = 0.0, 0.0, 0.0, 0.0

    train_loader = tqdm(train_loader, desc="🛠️ -- Training", leave=False)

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        if labels.dim() == 4:
            labels = labels.squeeze(1).to(device)
        else:
            labels = labels.to(device)

        if torch.isnan(images).any() or torch.isinf(images).any():
            raise ValueError("Input tensor contains NaN or Inf values.")
        if torch.isnan(labels).any() or torch.isinf(labels).any():
            raise ValueError("Input tensor contains NaN or Inf values.")
        
        with autocast(device_type='cuda'):
            optimizer.zero_grad()

            predictions = model(images)
            if torch.isnan(predictions).any() or torch.isinf(predictions).any():
                raise ValueError("Input tensor contains NaN or Inf values.")
            mean_loss, et_loss, tc_loss, wt_loss = loss_func(predictions, labels)

            scaler.scale(mean_loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss += mean_loss.item()
        total_et_loss += et_loss.item()
        total_tc_loss += tc_loss.item()
        total_wt_loss += wt_loss.item()

    avg_loss = total_loss / len(train_loader)
    avg_et_loss = total_et_loss / len(train_loader)
    avg_tc_loss = total_tc_loss / len(train_loader)
    avg_wt_loss = total_wt_loss / len(train_loader)

    return avg_loss, avg_et_loss, avg_tc_loss, avg_wt_loss


def validate_one_epoch(model, metric, val_loader, loss_func, device):
    """
    验证一个epoch
    :param model: 模型
    :param metric: 评估指标
    :param val_loader: 验证数据加载器
    :param loss_func: 损失函数
    :param device: 设备
    :return: 平均损失值和评估指标
    """
    metric_results = np.zeros((7, 4))
    model.eval()
    total_loss, total_et_loss, total_tc_loss, total_wt_loss = 0.0, 0.0, 0.0, 0.0

    with torch.no_grad():  # 关闭梯度计算
        val_loader = tqdm(val_loader, desc="🧐 -- Validating", leave=False)

        for batch_idx, (images, labels) in enumerate(val_loader):
            images, labels = images.to(device), labels.to(device)

            with autocast(device_type='cuda'):
                predictions = model(images)
                mean_loss, et_loss, tc_loss, wt_loss = loss_func(predictions, labels)
                batch_metrics = metric.update(predictions, labels)
                metric_results += batch_metrics

            total_loss += mean_loss.item()
            total_et_loss += et_loss.item()
            total_tc_loss += tc_loss.item()
            total_wt_loss += wt_loss.item()

    avg_loss = total_loss / len(val_loader)
    avg_et_loss = total_et_loss / len(val_loader)
    avg_tc_loss = total_tc_loss / len(val_loader)
    avg_wt_loss = total_wt_loss / len(val_loader)
    metric_results = metric_results / len(val_loader)

    return avg_loss, avg_et_loss, avg_tc_loss, avg_wt_loss, metric_results