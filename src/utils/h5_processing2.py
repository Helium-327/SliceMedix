# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2024/12/28 12:38:18
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: 获取多模态h5数据
=================================================
'''
import os
import nibabel as nib
import numpy as np
import h5py
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor


def get_nii_paths(dataset_path):
    """1. 获取数据集路径下的所有nii文件路径
    Args:
        dataset_path (str): 数据集路径
    Returns:
        list: nii文件路径列表
    """
    for dirs, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.nii.gz'):
                # print(file)
                yield os.path.join(dirs, file)  # yield 返回一个生成器，每次调用next()时返回下一个元素


def get_h5_paths(path):
    """1. 获取数据集路径下的所有h5文件路径
    Args:
        path (str): 数据集路径
    Returns:
        list: h5文件路径列表
    """
    return [os.path.join(root, name) for root, _, files in os.walk(path) for name in files if name.endswith(".h5")]


def save_nii_paths(dataset_path, nii_path, idx=False, id_path=None):
    """2. 保存数据集路径下的所有nii文件路径到文本文件中
    Args:
        dataset_path (str): 数据集路径
        nii_path (str): 保存的文本文件路径
    """
    generator = get_nii_paths(dataset_path)

    with open(nii_path, 'w') as f:
        for path in generator:
            f.write(path + '\n')

    print(f"✨✨  nii_paths from {dataset_path} saved to {nii_path}")

    if idx:
        save_nii_ids(dataset_path, id_path)

def save_nii_ids(dataset_path, id_path):
    generator = get_nii_paths(dataset_path)
    ids = [get_data_idx(path) for path in generator]
    ids = list(set(ids))
    with open(id_path, 'w') as f:
        for id in ids:
            f.write(id + '\n')
    print(f"✨✨  nii_ids from {dataset_path} saved to {id_path}")

def saved_h5_paths(h5_saved_dir, txt_saved_dir):
    
    paths = get_h5_paths(h5_saved_dir)

    with open(os.path.join(txt_saved_dir, f'multi_h5_paths.txt'), 'w') as f:
        for path in paths:
            f.write(path + '\n')

    print(f"✨ multi_h5_paths.txt saved to {os.path.join(txt_saved_dir, f'multi_h5_paths.txt')}")


def get_file_context_list(file_path):

    def readlines(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            while True:
                line = f.readline().strip("\n")
                if not line:
                    break
                else:
                    yield line

    reader = readlines(file_path)
    context = [line for line in reader]
    return context


def get_data_modal(nii_path, modal):
    return nii_path.split('/')[-1].split('.')[0].split('_')[-1] # 'flair'

def get_data_idx(nii_path):
    # /mnt/d/AI_Research/WS-Hub/Wsl-MedSeg/SliceMedix/data/raw/brats2021/BraTS2021_Training_Data/BraTS2021_00567/BraTS2021_00567_flair.nii.gz
    return nii_path.split('/')[-1].split('.')[0].split('_')[-2]  # '00567'

def load_nii_data(nii_path):
    """读取nii文件数据
    Args:
        nii_path (str): nii文件路径
    Returns:
        np.ndarray: nii文件数据
    """
    return nib.load(nii_path).get_fdata()



def get_modals_data_dict(datasets_root, id, modal_list):
    # 1. 获取多模态文件的文件路径
    # 2. 读取每个模态文件数据
    # 3. 合并多模态数据
    # 4. 返回合并后的数据
    data_path = os.path.join(datasets_root, f"BraTS2021_{id}".strip('\n'),)
    paths_dict = {}
    data_dict = {}
    for m in modal_list:
        paths_dict[m] = os.path.join(data_path, f"BraTS2021_{id}_{m}.nii.gz".strip('\n'))
        data_dict[m] = load_nii_data(paths_dict[m])
    
    return data_dict

    
def save_each_slice_to_h5(data, mask, h5_path_dir, h5_filename):
    if not os.path.exists(h5_path_dir):
        os.makedirs(h5_path_dir, exist_ok=True)
        print("The directory does not exist. But it has been created.")

    for i in tqdm(range(data.shape[-1])):
        slice_data = data[...,i]
        slice_mask = mask[...,i]

        # 保存到h5文件中
        h5_file_path = os.path.join(h5_path_dir, f"{h5_filename}_{i}th_slice.h5")
        with h5py.File(h5_file_path, 'w') as h5_file:
            h5_file.create_dataset('data', data=slice_data, compression='gzip')
            h5_file.create_dataset('mask', data=slice_mask, compression='gzip')
            # print(f"✨✨  {i}th slice saved to {h5_file_path}")

def process_id(id, datasets_path, modals, data_modals, h5_save_dir):
    data_dict = get_modals_data_dict(datasets_path, id, modals)
    data_list = [data_dict[m] for m in data_modals]
    data = np.stack(data_list, axis=-1)
    mask = data_dict['seg']
    save_each_slice_to_h5(data, mask, h5_save_dir, f'brats21_multi_{id}')




def main():
    modals = ['t1', 't1ce', 't2', 'flair', 'seg']
    data_modals = ['t1', 't1ce', 't2', 'flair']
    datasets_path = '/mnt/d/AI_Research/WS-Hub/Wsl-MedSeg/SliceMedix/data/raw/brats2021/BraTS2021_Training_Data'
    nii_paths = '/mnt/d/AI_Research/WS-Hub/Wsl-MedSeg/SliceMedix/data/nii_paths.txt'
    id_paths = '/mnt/d/AI_Research/WS-Hub/Wsl-MedSeg/SliceMedix/data/nii_idx.txt'

    h5_save_dir = '/root/workspace/MedSeg/SliceMedix/data/processed/brats21_h5_multi'

    # save_nii_paths(datasets_path, nii_paths, idx=True, id_path=id_paths)

    # paths = get_file_context_list(nii_paths)
    # ids = get_file_context_list(id_paths)

    # for id in tqdm(ids):
    #     data_dict = get_modals_data_dict(datasets_path, id, modals)
    #     data_list = [data_dict[m] for m in data_modals]
    #     data = np.stack(data_list, axis=0)
    #     mask = data_dict['seg']
    #     save_each_slice_to_h5(data, mask, h5_save_dir, f'brats21_multi_{id}')
    # 使用 ThreadPoolExecutor 创建一个线程池
    # with ThreadPoolExecutor(max_workers=8) as executor:
    #     # 使用 map 函数将 process_id 函数应用到每个 id 上
    #     list(tqdm(executor.map(process_id, ids, [datasets_path]*len(ids), [modals]*len(ids), [data_modals]*len(ids), [h5_save_dir]*len(ids)), total=len(ids)))
    data_dir = '/mnt/d/AI_Research/WS-Hub/Wsl-MedSeg/SliceMedix/data'
    saved_h5_paths(h5_save_dir, data_dir)

if __name__ == "__main__":
    main()
        

