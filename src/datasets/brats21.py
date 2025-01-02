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

# from transforms import *
from torch.utils.data import DataLoader

class BRATS21_2D(Dataset):
    def __init__(self, txt_file: str, transform=None, local_train=False, length=None):
        super().__init__()
        """
        Args:
            txt_file (str): 数据文件路径
            transform (callable, optional): 数据变换函数
            local_train (bool, optional): 是否只加载少量数据
            length (int, optional): 加载的局部数据量
        """
        assert os.path.exists(txt_file), f"{txt_file} does not exist"\
        
        # 
        self.txt_file = txt_file
        self.transform = transform

        # 读取少量数据
        self.local_train = local_train
        self.length = length

        # 获取数据路径列表
        self.data_paths_list = self.get_data_list() if not self.local_train else self.get_data_list()[:length]

    def nan_check(self, data):
        if isinstance(data, torch.Tensor):
            if torch.isnan(data).any() or torch.isinf(data).any():
                raise ValueError("Input tensor contains NaN or Inf values.")
        elif isinstance(data, np.ndarray):
            if np.isnan(data).any() or np.isinf(data).any():
                raise ValueError("Input tensor contains NaN or Inf values.")
            
    def load_data(self, h5_path: str) -> tuple:
        assert os.path.exists(h5_path), f"{h5_path} does not exist"
        try:
            with h5py.File(h5_path, 'r') as hf:
                data = hf['data'][:]
                mask = hf['mask'][:]
                mask[mask == 4] = 3
            # 检查数据中是否存在NaN或Inf值
            self.nan_check(data)
            self.nan_check(mask)

        except Exception as e:
            raise RuntimeError(f"Error loading data from {h5_path}: {e}")

        data, mask = self.preprocess_data(data, mask)

        return data.to(torch.float32), mask.to(torch.long)

    def preprocess_data(self, data, mask):
        # 数据预处理逻辑
        if self.transform:
            data, mask = self.transform(data, mask)
            self.nan_check(data)
            self.nan_check(mask)
            
        return data, mask

    def get_data_list(self):
        paths = []
        with open(self.txt_file) as f:
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
    txt_file = '/root/workspace/SliceMedix/data/multi_h5_paths.txt'
    # dataset = BRATS21_2D(txt_file, local_train=True, length=10)
    # print(len(dataset))
    train_txt = '/root/workspace/SliceMedix/data/train_paths.txt'
    val_txt = '/root/workspace/SliceMedix/data/val_paths.txt'
    test_txt = '/root/workspace/SliceMedix/data/test_paths.txt'


    transforms = Compose([
        ToTensor(),
        RandomCrop(crop_size=(196, 196)),
        RandomHorizontalFlip(flip_prob=0.5),
        RandomVerticalFlip(flip_prob=0.5),
        RandomRotation((0, 360)),
        Normalize(),
    ])

    trans_trian = Compose([
            ToTensor(),
            RandomCrop(crop_size=(196, 196)),
            RandomHorizontalFlip(flip_prob=0.5),
            RandomVerticalFlip(flip_prob=0.5),
            RandomRotation((0, 360)),
            Normalize(),
    ])

    trans_val = Compose([
            ToTensor(),
            # RandomCrop(crop_size=(196, 196)),
            # RandomHorizontalFlip(flip_prob=0.5),
            # RandomVerticalFlip(flip_prob=0.5),
            # RandomRotation((0, 360)),
            Normalize(),
    ])
    train_datasets = BRATS21_2D(train_txt,
                                transform=trans_trian)

    val_databsets = BRATS21_2D(val_txt, 
                                transform=trans_val)

    test_datasets = BRATS21_2D(test_txt,
                                transform=trans_val)
    
    train_loader = DataLoader(train_datasets, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_databsets, batch_size=16, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_datasets, batch_size=16, shuffle=False, num_workers=4)


    print(len(train_datasets))
    print(train_datasets[100][0].shape,  train_datasets[100][0].dtype)
    print(train_datasets[100][1].shape,  train_datasets[100][1].dtype)

    for i, (data, label) in enumerate(train_loader):
        print(i, data.shape, label.shape)
        print(data.dtype, label.dtype)
        break



