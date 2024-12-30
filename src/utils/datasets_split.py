# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2024/12/27 14:09:16
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: 
=================================================
'''
import os

def get_path_list(dataset_path):
    paths = []

    with open(dataset_path, 'r') as f:
        for line in f:
            path = line.strip().split()
            paths.append(path)

    return paths

def split_dataset(dataset_path, split_ratio=[0.8, 0.1, 0.1]):
    assert sum(split_ratio) == 1

    train_ratio, val_ratio, test_ratio = split_ratio

    paths = get_path_list(dataset_path)

    train_paths = paths[:int(len(paths) * train_ratio)]
    val_paths = paths[int(len(paths) * train_ratio):int(len(paths) * (train_ratio + val_ratio))]
    test_paths = paths[int(len(paths) * (train_ratio + val_ratio)):]

    return train_paths, val_paths, test_paths

def write_paths_to_file(paths, file_path):
    with open(file_path, 'w') as f:
        for path in paths:
            f.write(' '.join(path) + '\n')

    f.close()

if __name__ == '__main__':
    data_paths = '/mnt/d/AI_Research/WS-Hub/Wsl-MedSeg/SliceMedix/data/h5_paths.txt'
    train_paths, val_paths, test_paths = split_dataset(data_paths, [0.8, 0.1, 0.1])

    write_paths_to_file(train_paths, os.path.join(os.path.dirname(data_paths), 'train_paths.txt'))
    write_paths_to_file(val_paths, os.path.join(os.path.dirname(data_paths), 'val_paths.txt'))
    write_paths_to_file(test_paths, os.path.join(os.path.dirname(data_paths), 'test_paths.txt'))

    print("Done")

    