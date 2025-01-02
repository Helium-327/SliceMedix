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

    def __call__(self, data, mask):
        assert len(data.shape) == 3, "data must be a 3D array"
        _, h, w = data.shape
        crop_h, crop_w = self.crop_size
        start_h = np.random.randint(0, h - crop_h) if crop_h < h else 0
        start_w = np.random.randint(0, w - crop_w) if crop_w < w else 0

        # 随机裁剪
        data = data[:, start_h:start_h + crop_h, start_w:start_w + crop_w]
        mask = mask[:, start_h:start_h + crop_h, start_w:start_w + crop_w]

        return data, mask

class CenterRandomCrop(object):
    def __init__(self, crop_size, max_offset=10):
        """
        初始化中心随机裁剪类。
        :param crop_size: 裁剪尺寸，格式为 (crop_height, crop_width)
        :param max_offset: 最大偏移量，裁剪中心可以在图像中心附近随机偏移的最大像素数
        """
        self.crop_size = crop_size
        self.max_offset = max_offset

    def __call__(self, data, mask):
        """
        对输入数据和掩码进行中心随机裁剪。
        :param data: 输入数据，形状为 (C, H, W)
        :param mask: 输入掩码，形状为 (H, W) 或 (C, H, W)
        :return: 裁剪后的数据和掩码
        """
        assert len(data.shape) == 3, "data must be a 3D array"
        _, h, w = data.shape
        crop_h, crop_w = self.crop_size

        # 计算裁剪区域的中心点
        center_h = h // 2
        center_w = w // 2

        # 在中心点附近随机偏移
        offset_h = np.random.randint(-self.max_offset, self.max_offset + 1)
        offset_w = np.random.randint(-self.max_offset, self.max_offset + 1)

        # 计算裁剪区域的起始点
        start_h = max(0, center_h - crop_h // 2 + offset_h)
        start_w = max(0, center_w - crop_w // 2 + offset_w)

        # 确保裁剪区域不超出图像范围
        start_h = min(start_h, h - crop_h)
        start_w = min(start_w, w - crop_w)

        # 裁剪数据
        data = data[:, start_h:start_h + crop_h, start_w:start_w + crop_w]
        mask = mask[:, start_h:start_h + crop_h, start_w:start_w + crop_w]

        return data, mask

class Normalize(object):
    def __init__(self, mean=(0.114, 0.090, 0.170, 0.096), std=(0.199, 0.151, 0.282, 0.174)):
        self.mean = mean
        self.std = std

    def __call__(self, image, label):
        normalize = T.Normalize(mean=self.mean, std=self.std)
        image = normalize(image)
        for k in range(image.shape[0]):
            image[k] = (image[k] - image[k].min()) / (image[k].max() - image[k].min())
        return image, label

class Normalize2(object):
    def __init__(self):
        pass

    def __call__(self, image, label):
        for k in range(image.shape[0]):
            image[k] = (image[k] - image[k].min()) / (image[k].max() - image[k].min())
        
        return image, label

class Normalize3(object):
    def __init__(self):
        pass
    
    def __call__(self, image, label):
        for k in range(4):                   # 单个案例数据归一化，四个模态分别进行归一化
            x = image[k,...]                 # 取第k个模态的数据
            y = x[label]                      # 只取前景区域数据，，背景数据不包含其中
            x[label] -= y.mean()              # 仅对前景区域进行标准化处理
            x[label] /= (y.std() + 1e-6)
            # print(x[mask].mean(),x[mask].std())
            image[k,...] = x
        return image, label

if __name__ == "__main__":
    input_tensor = np.random.rand(240, 240, 4)
    label_tensor = np.random.rand(240, 240, 1)
    transforms = Compose([
        ToTensor(),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        RandomRotation(degrees=(0, 360)),
        RandomCrop(crop_size=(196, 196)),
        Normalize(),
    ])

    output_tensor, label_tensor = transforms(input_tensor, label_tensor)
    print(output_tensor.shape)
    print(label_tensor.shape)

