# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2024/12/18 14:20:45
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: 多模态体积数据h5格式
=================================================
'''

import nibabel as nib
import h5py
import numpy as np
import os
from matplotlib import pyplot as plt
from tqdm import tqdm

def processing_from_nii_to_h5(dataset_path, h5_save_dir):
    """将nii文件转换为h5文件并保存到指定路径
    Args:
        dataset_path (str): 数据集根目录路径
    Returns:
        None
    """
    paths, _ = get_dataset_list(dataset_path)
    
    for path in tqdm(paths):
        data_path = path.strip()
        mask_path = path.replace(path.split('/')[-1].split('_')[-1].split('.')[0], 'seg')
        h5_filename = data_path.split('/')[-1].replace('.nii.gz', '.h5')
        h5_path = h5_save_dir + h5_filename

        nii_to_h5(data_path, mask_path, h5_path)
    print("✨Done!!!")

def get_dataset_list(dataset_path):
    """获取数据集中的所有nii文件路径
    Args:
        dataset_path (str): 数据集根目录路径
    Returns:
        data_path_list (list): nii数据路径列表
        mask_path_list (list): nii标签路径列表
    """
    data_path_list = []
    mask_path_list = []

    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.nii.gz'):
                modal = file.split('/')[-1].split('_')[-1].split('.')[0]
                # idx = file.split('/')[-1].split('_')[-2]
                if modal == 'seg':
                    mask_path_list.append(os.path.join(root, file))
                else:
                    data_path_list.append(os.path.join(root, file))
    

    return data_path_list, mask_path_list


def nii_to_h5(data_path, mask_path, h5_path):
    """将nii文件转换为h5文件并保存到指定路径
    Args:
        data_path (str): nii文件的路径
        mask_path (str): nii文件的路径
        h5_path (str): h5文件的保存路径
    Returns:
        None
    """
    # 读取nii文件
    data_img = nib.load(data_path)
    mask_img = nib.load(mask_path)

    # 获取nii文件数据
    nii_data = data_img.get_fdata()
    nii_mask = mask_img.get_fdata()

    # 创建h5文件并保存数据
    if os.path.exists(h5_path):
        print(f"{h5_path} already exists")
    else:
        with h5py.File(h5_path, 'w') as hf:
            hf.create_dataset('data', data=nii_data, compression='gzip')
            hf.create_dataset('mask', data=nii_mask, compression='gzip')


def get_h5_paths(h5_saved_dir, filename):
    h5_names = os.listdir(h5_saved_dir)
    h5_dir = ('/').join(h5_saved_dir.split('/')[:])
    h5_list = [os.path.join(h5_dir, name) for name in h5_names]
    file_path = ('/').join(h5_saved_dir.split('/')[:-2])+'/'+filename
    print(h5_list)
    with open(file_path, 'w') as f:
        for h5_path in h5_list:
            f.write(h5_path+'\n')
    print(f"✨ {filename} Saved at {file_path}!!!")

def get_data_from_h5(h5_path):
    """从h5文件中提取数据
    Args:
        h5_path (str): h5文件的路径
    Returns:
        tuple: 包含数据和掩膜的元组，分别是ndarray类型
    """
    with h5py.File(h5_path, 'r') as hf:
        data = hf['data'][:]
        mask = hf['mask'][:]
    return data, mask

def plot_data_hwc(data, mask, Axis=2, num_col=10, single_size=3, save_fig=False):
    """将数据按照指定轴进行切片并绘制
    Args:
        data (ndarray): 数据，shape为HWC
        mask (ndarray): 掩膜，shape为HWC
        Axis (int): 切片的轴，默认为2
        num_col (int): 每行显示的切片数量，默认为10
        single_size (float): 每个子图的大小，默认为3
    Returns:
        None
    """
    num = data.shape[Axis]
    num_row = int(num / num_col) + 1

    fig, axs = plt.subplots(num_row, num_col, figsize=(single_size*num_col, single_size*num_row))


    for i in range(num_row):
        for j in range(num_col):

            idx = i*num_col + j
            if idx < num:
                if Axis == 2:
                    axs[i, j].imshow(data[:, :, idx], cmap='gray')
                    masked_mask = np.ma.masked_equal(mask[:, :, idx], 0)
                    axs[i, j].imshow(masked_mask, cmap='jet', alpha=0.5)
                elif Axis == 1:
                    axs[i, j].imshow(data[:, idx, :], cmap='gray')
                    masked_mask = np.ma.masked_equal(mask[:, idx, :], 0)
                    axs[i, j].imshow(masked_mask, cmap='jet', alpha=0.5)
                elif Axis == 0:
                    axs[i, j].imshow(data[idx, :, :], cmap='gray')
                    masked_mask = np.ma.masked_equal(mask[idx, :, :], 0)
                    axs[i, j].imshow(masked_mask, cmap='jet', alpha=0.5)
                else:
                    raise ValueError('Axis must be 0,1 or 2')
                axs[i, j].set_title(f"Axis:{Axis}; idx:{idx}")
                axs[i, j].axis('off')

    plt.tight_layout()
    if save_fig:
        plt.savefig("slice_plot.jpg", dpi=600)
    plt.show()

if __name__ == "__main__":

    # 1. 转换nii文件到h5文件并保存
    dataset_path = '/mnt/d/AI_Research/WS-Hub/Wsl-MedSeg/SliceMedix/data/raw/brats2021'
    h5_save_dir = '/mnt/d/AI_Research/WS-Hub/Wsl-MedSeg/SliceMedix/data/processed/brats21_h5'
    # processing_from_nii_to_h5(dataset_path, h5_save_dir)

    # TODO: 2. 获取所有的h5文件路径，并保存文件 
    get_h5_paths(h5_save_dir, 'h5_paths.txt')

    # TODO: 3. 读取h5文件并提取数据
    h5_path = '/mnt/d/AI_Research/WS-Hub/Wsl-MedSeg/SliceMedix/data/processed/brats21_h5/BraTS2021_00344_t1ce.h5'

    data, mask = get_data_from_h5(h5_path)

    # TODO: 4. 可视化数据
    plot_data_hwc(data=data, mask=mask, Axis=2, save_fig=True)
