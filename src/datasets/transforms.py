# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2024/12/27 16:59:45
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: 二维数据数据增强，形状：(155, 240, 240)
=================================================
'''

import torch
import numpy as np
from torchvision import transforms as T
from torchvision.transforms import functional as F

seed = 42                       #seed必须是int，可以自行设置
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)    #让显卡产生的随机数一致
torch.cuda.manual_seed_all(seed)#多卡模式下，让所有显卡生成的随机数一致？这个待验证
np.random.seed(seed)            #numpy产生的随机数一致


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data, mask):
        for t in self.transforms:
            data, mask = t(data, mask)

        return data, mask

class ToTensor(object):
    """Convert ndarrays in sample to Tensors.
    1. 转换通道顺序
    2. 转换为torch tensor
    """

    def __call__(self, data, mask):
        """
        numpy image: H x W x C
        torch image: C X H X W

        """
        if type(data) is not np.ndarray:
            raise TypeError("Expected numpy array, got {}".format(type(data)))
        # else:
        #     data = data.transpose((2, 0, 1))
        #     mask = mask.transpose((2, 0, 1))

        data = F.to_tensor(data).to(torch.float32) 
        mask = F.to_tensor(mask).to(torch.int64)

        return data, mask
    
# 角度增强
class RandomHorizontalFlip(object):
    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob
        
    def __call__(self, data, mask):
        data = T.RandomHorizontalFlip(p=self.flip_prob)(data)
        mask = T.RandomHorizontalFlip(p=self.flip_prob)(mask)
        
        return data, mask
    
class RandomVerticalFlip(object):
    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, data, mask):
        data = T.RandomVerticalFlip(p=self.flip_prob)(data)
        mask = T.RandomVerticalFlip(p=self.flip_prob)(mask)

        return data, mask

class RandomRotation(object):
    def __init__(self, degrees):
        self.degrees = degrees

    def __call__(self, data, mask):
        data = T.RandomRotation(self.degrees)(data)
        mask = T.RandomRotation(self.degrees)(mask)

        return data, mask
            
# 尺寸增强
class RandomCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size
    
    def RandomCrop(self, data, mask):
        assert data.shape == mask.shape, "data and mask must have the same shape"
        assert len(data.shape) == 3, "data must be a 3D array"

        # 随机裁剪
        d, h, w = data.shape
        crop_d, crop_h, crop_w = self.crop_size
        start_h = np.random.randint(0, h - crop_h) if crop_h < h else 0
        start_w = np.random.randint(0, w - crop_w) if crop_w < w else 0
        start_d = (d - crop_d) // 2

        data_cropped = data[start_d:start_d+crop_d, start_w:start_w+crop_w, start_h:start_h+crop_h]
        mask_cropped = mask[start_d:start_d+crop_d, start_w:start_w+crop_w, start_h:start_h+crop_h]

        return data_cropped, mask_cropped

    def __call__(self, data, mask):
        return self.RandomCrop(data, mask)


class Normalize(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data, mask):
        data = T.Normalize(mean=self.mean, std=self.std)(data)

        return data, mask


if __name__ == "__main__":
    input_tensor = np.random.rand(240, 240, 155)
    label_tensor = np.random.rand(240, 240, 155)
    transforms = Compose([
        ToTensor(),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        RandomRotation(degrees=(0, 360)),
        RandomCrop(crop_size=(144, 240, 240)),
        Normalize(),
    ])

    output_tensor, label_tensor = transforms(input_tensor, label_tensor)
    print(output_tensor.shape)
    print(label_tensor.shape)

