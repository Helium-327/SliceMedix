# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2024/12/30 14:37:09
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: Metrics for evaluation
=================================================
'''

import torch
import torch.nn.functional as F
import numpy as np

class EvaluationMetrics:
    def __init__(self, smooth=1e-5, num_classes=4):
        self.smooth = smooth
        self.num_classes = num_classes
        self.sub_areas = ['ET', 'TC', 'WT']

    def pre_processing(self, y_pred, y_label):
        """
        预处理：
            1.挑选出预测概率最大的类别；
            2.one-hot处理
        :param y_pred: 预测标签 (B, C, H, W)
        :param y_label: 真实标签 (B, C, H, W)
        :return: 处理后的预测标签和真实标签
        """
        assert y_pred.dim() == 4 and y_label.dim() == 4, "输入张量必须是4维的 (B, C, H, W)"
        
        et_pred = y_pred[:, 3, ...]
        tc_pred = y_pred[:, 1, ...] + y_pred[:, 3, ...]
        wt_pred = y_pred[:, 1:, ...].sum(dim=1)
        
        et_mask = y_label[:, 3, ...]
        tc_mask = y_label[:, 1, ...] + y_label[:, 3, ...]
        wt_mask = y_label[:, 1:, ...].sum(dim=1)
        
        pred_list = [et_pred, tc_pred, wt_pred]
        mask_list = [et_mask, tc_mask, wt_mask]
        
        return pred_list, mask_list

    def culculate_confusion_matrix(self, y_pred, y_label):
        """
        计算混淆矩阵元素值
        :param y_pred: 预测标签 (B, H, W)
        :param y_label: 真实标签 (B, H, W)
        :return: 混淆矩阵元素值TP、FN、 FP、TN
        """
        assert y_pred.dim() == 3 and y_label.dim() == 3, "输入张量必须是3维的 (B, H, W)"
        
        tensor_one = torch.tensor(1, device=y_pred.device)
        
        TP = (y_pred * y_label).sum(dim=(-2, -1))  # 预测为正类，实际也为正类
        FN = ((tensor_one - y_pred) * y_label).sum(dim=(-2, -1))  # 预测为负类，实际为正类
        FP = (y_pred * (tensor_one - y_label)).sum(dim=(-2, -1))  # 预测为正类，实际为负类
        TN = ((tensor_one - y_pred) * (tensor_one - y_label)).sum(dim=(-2, -1))  # 预测为负类，实际也为负类
        
        return TP, FN, FP, TN

    def dice_coefficient(self, y_pred, y_label):
        """
        计算Dice 系数
        :param y_pred: 预测标签 (B, C, H, W)
        :param y_label: 真实标签 (B, C, H, W)
        :return: Dice 系数
        """
        assert y_pred.dim() == 4 and y_label.dim() == 4, "输入张量必须是4维的 (B, C, H, W)"
        
        
        y_pred = torch.argmax(y_pred, dim=1).to(dtype=torch.int64)  # 降维，选出概率最大的类索引值
        y_pred = F.one_hot(y_pred, num_classes=4).permute(0, 3, 1, 2).float()  # one-hot
        # y_label = F.one_hot(y_label, num_classes=4).permute(0, 3, 1, 2).float()  # one-hot

        if len(y_label.unique()) == 4:
            y_label = F.one_hot(y_label.squeeze(1), self.num_classes).permute(0, 3, 1, 2) 
        else:
            y_label_zeros = torch.zeros_like(y_pred).long()
            y_label_zeros.scatter_(1, y_label.unsqueeze(1), 1)
            y_label = y_label_zeros.clone()
        
        pred_list, mask_list = self.pre_processing(y_pred, y_label)  # 获取子区的预测标签和真实标签
        
        dice_coeffs = {}
        for sub_area, sub_pred, sub_mask in zip(self.sub_areas, pred_list, mask_list):
            intersection = (sub_pred * sub_mask).sum(dim=(-2, -1))
            union = sub_pred.sum(dim=(-2, -1)) + sub_mask.sum(dim=(-2, -1))
            dice_c = (2. * intersection + self.smooth) / (union + self.smooth)
            dice_coeffs[sub_area] = dice_c.mean()
        
        intersection = (y_pred * y_label).sum(dim=(-2, -1))
        union = y_pred.sum(dim=(-2, -1)) + y_label.sum(dim=(-2, -1))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_coeffs['global mean'] = dice.mean()

        et_dice = dice_coeffs['ET'].item()
        tc_dice = dice_coeffs['TC'].item()
        wt_dice = dice_coeffs['WT'].item()
        mean_dice = (et_dice + tc_dice + wt_dice) / 3
        global_mean_dice = dice_coeffs['global mean'].item()
        
        return mean_dice, et_dice, tc_dice, wt_dice

    def jaccard_index(self, y_pred, y_label):
        """
        计算Jaccard 系数
        :param y_pred: 预测标签 (B, C, H, W)
        :param y_label: 真实标签 (B, C, H, W)
        :return: Jaccard 系数 (全局平均)
        """
        assert y_pred.dim() == 4 and y_label.dim() == 4, "输入张量必须是4维的 (B, C, H, W)"
        
        y_pred = torch.argmax(y_pred, dim=1).to(dtype=torch.int64)  # 降维，选出概率最大的类索引值
        y_pred = F.one_hot(y_pred, num_classes=4).permute(0, 3, 1, 2).float()  # one-hot
        # y_label = F.one_hot(y_label, num_classes=4).permute(0, 3, 1, 2).float()  # one-hot
        if len(y_label.unique()) == 4:
            y_label = F.one_hot(y_label.squeeze(1), self.num_classes).permute(0, 3, 1, 2) 
        else:
            y_label_zeros = torch.zeros_like(y_pred).long()
            y_label_zeros.scatter_(1, y_label.unsqueeze(1), 1)
            y_label = y_label_zeros.clone()
        
        
        pred_list, mask_list = self.pre_processing(y_pred, y_label) 
        jaccard_coeffs = {}

        for sub_area, pred, mask in zip(self.sub_areas, pred_list, mask_list):
            intersection = (pred * mask).sum(dim=(-2, -1))
            union = pred.sum(dim=(-2, -1)) + mask.sum(dim=(-2, -1)) - intersection
            jaccard = (intersection + self.smooth) / (union + self.smooth)
            jaccard_coeffs[sub_area] = jaccard.mean()

        et_jaccard = jaccard_coeffs['ET'].item()
        tc_jaccard = jaccard_coeffs['TC'].item()
        wt_jaccard = jaccard_coeffs['WT'].item()
        mean_jaccard = (et_jaccard + tc_jaccard + wt_jaccard) / 3
        
        return mean_jaccard, et_jaccard, tc_jaccard, wt_jaccard

    def recall(self, y_pred, y_label):
        """
        计算Recall(查全率，敏感性（真阳性率））
        :param y_pred: 预测标签 (B, C, H, W)
        :param y_label: 真实标签 (B, C, H, W)
        :return: 敏感性/特异性（全局平均 、ET 、TC、WT）
        """
        assert y_pred.dim() == 4 and y_label.dim() == 4, "输入张量必须是4维的 (B, C, H, W)"
        
        y_pred = torch.argmax(y_pred, dim=1).to(dtype=torch.int64)  # 降维，选出概率最大的类索引值
        y_pred = F.one_hot(y_pred, num_classes=4).permute(0, 3, 1, 2).float()  # one-hot
        # y_label = F.one_hot(y_label, num_classes=4).permute(0, 3, 1, 2).float()  # one-hot
        if len(y_label.unique()) == 4:
            y_label = F.one_hot(y_label.squeeze(1), self.num_classes).permute(0, 3, 1, 2) 
        else:
            y_label_zeros = torch.zeros_like(y_pred).long()
            y_label_zeros.scatter_(1, y_label.unsqueeze(1), 1)
            y_label = y_label_zeros.clone()
        
        pred_list, mask_list = self.pre_processing(y_pred, y_label) 
        recall_scores = {}

        for sub_area, pred, mask in zip(self.sub_areas, pred_list, mask_list):
            TP, FN, _, _ = self.culculate_confusion_matrix(pred, mask)
            recall = (TP + self.smooth) / (TP + FN + self.smooth)
            recall_scores[sub_area] = recall.mean()
        
        et_recall = recall_scores['ET'].item()
        tc_recall = recall_scores['TC'].item()
        wt_recall = recall_scores['WT'].item()
        mean_recall = (et_recall + tc_recall + wt_recall) / 3

        return mean_recall, et_recall, tc_recall, wt_recall

    def precision(self, y_pred, y_label):
        """
        计算Precision （查准率，特异性（真阴性率））
        :param y_pred: 预测标签 (B, C, H, W)
        :param y_label: 真实标签 (B, C, H, W)
        :return: 全局平均查准率， ET查准率，TC查准率，WT查准率
        """
        assert y_pred.dim() == 4 and y_label.dim() == 4, "输入张量必须是4维的 (B, C, H, W)"
        
        y_pred = torch.argmax(y_pred, dim=1).to(dtype=torch.int64)  # 降维，选出概率最大的类索引值
        y_pred = F.one_hot(y_pred, num_classes=4).permute(0, 3, 1, 2).float()  # one-hot
        # y_label = F.one_hot(y_label, num_classes=4).permute(0, 3, 1, 2).float()  # one-hot
        if len(y_label.unique()) == 4:
            y_label = F.one_hot(y_label.squeeze(1), self.num_classes).permute(0, 3, 1, 2) 
        else:
            y_label_zeros = torch.zeros_like(y_pred).long()
            y_label_zeros.scatter_(1, y_label.unsqueeze(1), 1)
            y_label = y_label_zeros.clone()
        
        pred_list, mask_list = self.pre_processing(y_pred, y_label) 
        precision_scores = {}

        for sub_area, pred, mask in zip(self.sub_areas, pred_list, mask_list):
            TP, _, FP, _ = self.culculate_confusion_matrix(pred, mask)
            precision = (TP + self.smooth) / (TP + FP + self.smooth)
            precision_scores[sub_area] = precision.mean()
        
        et_precision = precision_scores['ET'].item()
        tc_precision = precision_scores['TC'].item()
        wt_precision = precision_scores['WT'].item()
        mean_precision = (et_precision + tc_precision + wt_precision) / 3
        
        return mean_precision, et_precision, tc_precision, wt_precision

    def accuracy(self, y_pred, y_label):
        """
        准确率
        :param y_pred: 预测标签 (B, C, H, W)
        :param y_label: 真实标签 (B, C, H, W)
        :return: 准确率
        """
        assert y_pred.dim() == 4 and y_label.dim() == 4, "输入张量必须是4维的 (B, C, H, W)"
        
        y_pred = torch.argmax(y_pred, dim=1).to(dtype=torch.int64)  # 降维，选出概率最大的类索引值
        y_pred = F.one_hot(y_pred, num_classes=4).permute(0, 3, 1, 2).float()  # one-hot
        # y_label = F.one_hot(y_label, num_classes=4).permute(0, 3, 1, 2).float()  # one-hot
        if len(y_label.unique()) == 4:
            y_label = F.one_hot(y_label.squeeze(1), self.num_classes).permute(0, 3, 1, 2) 
        else:
            y_label_zeros = torch.zeros_like(y_pred).long()
            y_label_zeros.scatter_(1, y_label.unsqueeze(1), 1)
            y_label = y_label_zeros.clone()
        
        
        pred_list, mask_list = self.pre_processing(y_pred, y_label)
        accuracy_scores = {}

        for sub_area, pred, mask in zip(self.sub_areas, pred_list, mask_list):
            TP, FN, FP, TN = self.culculate_confusion_matrix(pred, mask)
            accuracy = (TP + TN) / (TP + FN + FP + TN)
            accuracy_scores[sub_area] = accuracy.mean()
        
        et_accuracy = accuracy_scores['ET'].item()
        tc_accuracy = accuracy_scores['TC'].item()
        wt_accuracy = accuracy_scores['WT'].item()
        mean_accuracy = (et_accuracy + tc_accuracy + wt_accuracy) / 3
        
        return mean_accuracy, et_accuracy, tc_accuracy, wt_accuracy

    def f1_score(self, y_pred, y_label):
        """
        计算F1值
        :param y_pred: 预测标签 (B, C, H, W)
        :param y_label: 真实标签 (B, C, H, W)
        :return: F1值
        """
        assert y_pred.dim() == 4 and y_label.dim() == 4, "输入张量必须是4维的 (B, C, H, W)"
        
        precision_list = self.precision(y_pred, y_label)
        recall_list = self.recall(y_pred, y_label)
        
        f1_scores = {}
        f1_scores['ET'] = 2 * (precision_list[1] * recall_list[1]) / (precision_list[1] + recall_list[1])
        f1_scores['TC'] = 2 * (precision_list[2] * recall_list[2]) / (precision_list[2] + recall_list[2])
        f1_scores['WT'] = 2 * (precision_list[3] * recall_list[3]) / (precision_list[3] + recall_list[3])
        
        et_f1 = f1_scores['ET']
        tc_f1 = f1_scores['TC']
        wt_f1 = f1_scores['WT']
        mean_f1 = (et_f1 + tc_f1 + wt_f1) / 3       
        
        return mean_f1, et_f1, tc_f1, wt_f1
    
    def f2_score(self, y_pred, y_label):         #FIXME: 正确性有待测试
        """
        计算F2值
        :param y_pred: 预测标签
        :param y_label: 真实标签
        :return: F2值
        """
        assert y_pred.dim() == 4 and y_label.dim() == 4, "输入张量必须是4维的 (B, C, H, W)"
        # pred_list, mask_list = self.pre_processing(y_pred, y_mask)
        precision_list = self.precision(y_pred, y_label)
        recall_list = self.recall(y_pred, y_label)
        
        f2_scores = {}
        # f1_score on ET
        f2_scores[self.sub_areas[0]] = (5*precision_list[1] * recall_list[1]) / (4*precision_list[1] + recall_list[1])
        # f1_socre on TC
        f2_scores[self.sub_areas[1]] = (5*precision_list[2] * recall_list[2]) / (4*precision_list[2] + recall_list[2])
        # f1_score on WT
        f2_scores[self.sub_areas[2]] = (5*precision_list[3] * recall_list[3]) / (4*precision_list[3] + recall_list[3])
        
        et_f2 = f2_scores['ET']
        tc_f2 = f2_scores['TC']
        wt_f2 = f2_scores['WT']
        mean_f2 = (et_f2 + tc_f2 + wt_f2) / 3

        return mean_f2, et_f2, tc_f2, wt_f2
    
    def update(self, y_pred, y_label):
        """
        更新评估指标
        :param y_pred: 预测标签 (B, C, H, W)
        :param y_label: 真实标签 (B, C, H, W)
        :return: 所有的评估指标
        """
        assert y_pred.dim() == 4 and y_label.dim() == 4, "输入张量必须是4维的 (B, C, H, W)"
        
        dice_scores = self.dice_coefficient(y_pred, y_label)
        jacc_scores = self.jaccard_index(y_pred, y_label)
        accuracy_scores = self.accuracy(y_pred, y_label)
        precision_scores = self.precision(y_pred, y_label)
        recall_scores = self.recall(y_pred, y_label)
        f1_scores = self.f1_score(y_pred, y_label)
        f2_scores = self.f2_score(y_pred, y_label)

        metrics = [dice_scores, jacc_scores, accuracy_scores, precision_scores, recall_scores, f1_scores, f2_scores]
        metrics = np.stack(metrics, axis=0)  # [7, 4]
        metrics = np.nan_to_num(metrics)
        return metrics

    def format_value(value, decimals=4):
        """返回一个格式化后的字符串，保留指定的小数位数"""
        return f"{value:.{decimals}f}"

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # local_root = ""