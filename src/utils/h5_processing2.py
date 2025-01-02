# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2024/12/28 12:38:18
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: 获取多模态切片数据集 h5文件
=================================================
'''
import os
import nibabel as nib
import numpy as np
import h5py
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


def get_nii_file_paths(dataset_path):
    """获取数据集路径下的所有 .nii.gz 文件路径"""
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.nii.gz'):
                yield os.path.join(root, file)  # 使用 yield 返回生成器，节省内存


def get_h5_file_paths(path):
    """获取数据集路径下的所有 .h5 文件路径"""
    return [os.path.join(root, name) for root, _, files in os.walk(path) for name in files if name.endswith(".h5")]


def save_nii_file_paths(dataset_path, nii_path):
    """保存 .nii.gz 文件路径到文本文件中"""
    with open(nii_path, 'w') as f:
        for path in get_nii_file_paths(dataset_path):
            f.write(path + '\n')  # 将每个路径写入文件
    print(f"✨✨  nii_paths from {dataset_path} saved to {nii_path}")


def save_nii_ids(dataset_path, id_path):
    """保存 .nii.gz 文件的 ID 到文本文件中"""
    ids = {extract_id_from_path(path) for path in get_nii_file_paths(dataset_path)}  # 使用集合去重
    with open(id_path, 'w') as f:
        for id in ids:
            f.write(id + '\n')  # 将每个 ID 写入文件
    print(f"✨✨  nii_ids from {dataset_path} saved to {id_path}")



def save_h5_file_paths(h5_saved_dir, txt_saved_dir):
    """保存 .h5 文件路径到文本文件中"""
    paths = get_h5_file_paths(h5_saved_dir)
    with open(os.path.join(txt_saved_dir, 'multi_h5_paths.txt'), 'w') as f:
        for path in paths:
            f.write(path + '\n')  # 将每个 .h5 文件路径写入文件
    print(f"✨ multi_h5_paths.txt saved to {os.path.join(txt_saved_dir, 'multi_h5_paths.txt')}")

def read_lines_from_file(file_path):
    """读取文本文件内容并返回列表"""
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]  # 去除空行并返回列表


def extract_modal_from_path(nii_path):
    """从 .nii.gz 文件路径中提取模态信息"""
    return os.path.basename(nii_path).split('.')[0].split('_')[-1]  # 提取文件名中的模态信息


def extract_id_from_path(nii_path):
    """从 .nii.gz 文件路径中提取数据 ID"""
    return os.path.basename(nii_path).split('.')[0].split('_')[-2]  # 提取文件名中的 ID


def load_nii_file_data(nii_path):
    """读取 .nii.gz 文件数据"""
    return nib.load(nii_path).get_fdata()  # 使用 nibabel 库读取 .nii.gz 文件数据


def get_multimodal_data_dict(datasets_root, id, modal_list):
    """获取多模态数据的字典"""
    data_path = os.path.join(datasets_root, f"BraTS2021_{id}".strip('\n'))  # 构建数据路径
    return {m: load_nii_file_data(os.path.join(data_path, f"BraTS2021_{id}_{m}.nii.gz".strip('\n'))) for m in modal_list}  # 返回多模态数据字典


def save_slices_to_h5(data, mask, h5_path_dir, h5_filename):
    """将每个切片保存为 .h5 文件"""
    os.makedirs(h5_path_dir, exist_ok=True)  # 创建目录（如果不存在）
    for i in tqdm(range(data.shape[-1])):  # 遍历每个切片
        slice_data = data[..., i]
        slice_mask = mask[..., i]
        if slice_mask.max() > 0:  # 只保存有标签的切片
            h5_file_path = os.path.join(h5_path_dir, f"{h5_filename}_{i}th_slice.h5")
            with h5py.File(h5_file_path, 'w') as h5_file:  # 创建 .h5 文件
                h5_file.create_dataset('data', data=slice_data, compression='gzip')  # 保存数据
                h5_file.create_dataset('mask', data=slice_mask, compression='gzip')  # 保存标签


def split_dataset_into_subsets(data_paths_txt, split_ratio=[0.8, 0.1, 0.1]):
    """将数据集划分为训练集、验证集和测试集"""
    data_paths_list = read_lines_from_file(data_paths_txt)  # 读取所有路径
    num_data = len(data_paths_list)
    train_list = data_paths_list[:int(num_data * split_ratio[0])]  # 训练集
    val_list = data_paths_list[int(num_data * split_ratio[0]): int(num_data * (sum(split_ratio[:2])))]  # 验证集
    test_list = data_paths_list[int(num_data * sum(split_ratio[:2])):]  # 测试集
    base_dir = os.path.dirname(data_paths_txt)  # 获取基础目录
    for subset, name in zip([train_list, val_list, test_list], ['train_paths.txt', 'val_paths.txt', 'test_paths.txt']):
        with open(os.path.join(base_dir, name), 'w') as f:
            for path in subset:
                f.write(path + '\n')  # 将每个子集的路径写入文件
        print(f"✨✨  {name} saved to {os.path.join(base_dir, name)}")


def process_multimodal_data_for_id(id, datasets_path, modals, data_modals, h5_save_dir):
    """处理单个 ID 的多模态数据并保存为 .h5 文件"""
    data_dict = get_multimodal_data_dict(datasets_path, id, modals)  # 获取多模态数据
    data = np.stack([data_dict[m] for m in data_modals], axis=-1)  # 合并多模态数据
    save_slices_to_h5(data, data_dict['seg'], h5_save_dir, f'brats21_multi_{id}')  # 保存切片为 .h5 文件


def main():
    """主函数，执行数据处理流程"""
    modals = ['t1', 't1ce', 't2', 'flair', 'seg']  # 所有模态
    data_modals = ['t1', 't1ce', 't2', 'flair']  # 数据模态（不包括标签）
    # datasets_path = '/mnt/d/AI_Research/WS-Hub/Wsl-MedSeg/SliceMedix/data/raw/brats2021/BraTS2021_Training_Data'  # 数据集路径
    # h5_save_dir = '/root/workspace/SliceMedix/data/processed/brats21_h5_multi'  # .h5 文件保存路径
    # ids = read_lines_from_file('/mnt/d/AI_Research/WS-Hub/Wsl-MedSeg/SliceMedix/data/nii_idx.txt')  # 读取所有 ID
    
    # # 使用线程池并行处理每个 ID 的数据
    # with ThreadPoolExecutor(max_workers=8) as executor:
    #     list(tqdm(executor.map(process_multimodal_data_for_id, ids, [datasets_path]*len(ids), [modals]*len(ids), [data_modals]*len(ids), [h5_save_dir]*len(ids)), total=len(ids)))
    
    # 保存 .h5 文件路径并划分数据集
    # save_h5_file_paths(h5_save_dir, '/mnt/d/AI_Research/WS-Hub/Wsl-MedSeg/SliceMedix/data')
    split_dataset_into_subsets('/root/workspace/SliceMedix/data/multi_h5_paths_select.txt')


if __name__ == "__main__":
    main()