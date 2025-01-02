# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2024/12/30 14:15:58
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: loss function
=================================================
'''

from tabnanny import check
import torch

import torch.nn.functional as F

class DiceLoss():
    def __init__(self, smooth = 1e-5):
        
        self.smooth = smooth
        self.sub_areas = ['ET', 'TC', 'WT']

        self.labels = {
            'BG': 0,
            'NCR': 1,
            'ED': 2,
            'ET': 3,
        }

        self.num_classes = len(self.labels)

    def __call__(self, y_pred, y_label):
        if torch.isnan(y_pred).any() or torch.isinf(y_pred).any():
            raise ValueError("Input tensor contains NaN or Inf values.")
        
        if torch.isnan(y_label).any() or torch.isinf(y_label).any():
            raise ValueError("Input tensor contains NaN or Inf values.")

        # 获取标签的最大值（#!计算loss时不能使用，因为argmax不可导，无法进行反向传播s）
        # y_label = torch.argmax(y_label, dim=1) 

        # one-hot encoding
        if len(y_label.unique()) == 4:
            y_label = F.one_hot(y_label.squeeze(1), self.num_classes).permute(0, 3, 1, 2).float()
        else:
            y_label_zeros = torch.zeros_like(y_pred).long()
            y_label_zeros.scatter_(1, y_label.unsqueeze(1), 1)
            y_label = y_label_zeros.clone().float()
        
        assert y_label.shape == y_pred.shape, "标签和预测的形状不一致"
        area_et_loss, area_tc_loss, area_wt_loss = self.get_every_subarea_loss(y_pred, y_label)

        mean_loss = (area_et_loss + area_tc_loss + area_wt_loss) / 3
        return mean_loss, area_et_loss, area_tc_loss, area_wt_loss
    

    def get_every_subarea_loss(self, y_pred, y_label):
    
        loss_dict = {}
        pred_list, mask_list = splitSubAreas(y_pred, y_label)

        for sub_area, pred, label in zip(self.sub_areas, pred_list, mask_list):
            intersection = (pred * label).sum(dim=(-2, -1))
            union = (pred + label).sum(dim=(-2, -1))
            # dice_c = (2. * intersection + self.smooth) / (union + self.smooth)
            dice_c = (2. * intersection) / (union + self.smooth)
            if torch.isnan(dice_c).any() or torch.isinf(dice_c).any():
                raise ValueError("Loss contains NaN values.")
            loss_dict[sub_area] = 1. - dice_c.mean()

        area_et_loss = loss_dict['ET']
        area_tc_loss = loss_dict['TC']
        area_wt_loss = loss_dict['WT']
        
        return area_et_loss, area_tc_loss, area_wt_loss


def splitSubAreas(y_pred, y_mask):
    """
    分割出子区域
    :param y_pred: 预测值 [batch, 4, W, H]
    :param y_mask: 真实值 [batch, 4, W, H]
    """
    et_pred = y_pred[:, 3,...]
    tc_pred = y_pred[:, 1,...] + y_pred[:,3,...]
    wt_pred = y_pred[:, 1:,...].sum(dim=1)
    
    et_mask = y_mask[:, 3,...]
    tc_mask = y_mask[:, 1,...] + y_mask[:,3,...]
    wt_mask = y_mask[:, 1:,...].sum(dim=1)
    
    pred_list = [et_pred, tc_pred, wt_pred]
    mask_list = [et_mask, tc_mask, wt_mask]
    return pred_list, mask_list


if __name__ == '__main__':
    y_pred = torch.randn(16, 4, 128, 128)
    # y_pred = torch.argmax(y_pred, dim=1)
    y_label = torch.randn(16, 128, 128)
    
    y_label = torch.where(torch.rand_like(y_label) < 0.5, torch.tensor(0), torch.tensor(2))
    
    y_label_one_hot = torch.zeros(16, 4, 128, 128)
    y_label_one_hot.scatter_(1, y_label.unsqueeze(1), 1)

    diceLossFunc = DiceLoss()

    print(f"DiceLoss : {diceLossFunc(y_pred, y_label)}")
    