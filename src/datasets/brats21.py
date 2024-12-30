# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2024/12/18 21:42:11
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: BRATS 2021 dataset loader
=================================================
'''

import os
import torch
import numpy as np
import h5py

import nibabel as nib
from torch.utils.data import Dataset

from torch.nn.functional import one_hot

from transforms import *


class BRATS21_2D(Dataset):
    def __init__(self, data_file, transform=None, local_train=False, length=None):
        super().__init__()
        """
        Args:
            data_file (str): 数据文件路径
            transform (callable, optional): 数据变换函数
            local_train (bool, optional): 是否只加载少量数据
            length (int, optional): 加载的局部数据量
        """
        assert os.path.exists(data_file), f"{data_file} does not exist"\
        
        # 
        self.data_file = data_file
        self.transform = transform

        # 读取少量数据
        self.local_train = local_train
        self.length = length

        # 获取数据路径列表
        self.data_paths_list = self.get_data_list() if not self.local_train else self.get_data_list()[:length]

            
    def load_data(self, h5_path):
        assert os.path.exists(h5_path), f"{h5_path} does not exist"

        # 从h5文件中读取数据
        with h5py.File(h5_path, 'r') as hf:
            data = hf['data'][:]
            mask = hf['mask'][:]

        # 数据预处理
        
        return data, mask

    def get_data_list(self):
        paths = []
        with open(self.data_file) as f:
            for line in f:
                path = line.strip()
                paths.append(path)
        return paths

    def __len__(self):
        return len(self.data_paths_list)

    def __getitem__(self, index):
        data, mask = self.load_data(self.data_paths_list[index])
        return data, mask



if __name__ == '__main__':
    data_file = '/mnt/d/AI_Research/WS-Hub/Wsl-MedSeg/SliceMedix/data/multi_h5_paths.txt'
    dataset = BRATS21_2D(data_file, local_train=True, length=10)
    print(len(dataset))

    transforms = Compose([
        ToTensor(),
        RandomHorizontalFlip(flip_prob=0.5),
        RandomVerticalFlip(flip_prob=0.5),
        RandomRotation((0, 360)),
    ])
    data, mask = dataset[0]
    print(data.shape, mask.shape)

    data, mask = transforms(data, mask)
    print(data.shape, mask.shape)

    train_dataset = BRATS21_2D(data_file, 
                                transform=transforms,
                                # local_train=True, 
                                # length=1
                                )   

    print(len(train_dataset))