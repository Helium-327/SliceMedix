# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2024/12/30 15:13:46
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: 配置脚本
*      VERSION: v1.0
=================================================
'''
import os
import json
import yaml
import torch
import torch.nn as nn
import argparse
from train import train
from tabulate import tabulate
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD, RMSprop, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.amp import GradScaler
from utils.logger_tools import *
from utils.shell_tools import *
from utils.tb_tools import *
from evaluate.metrics import EvaluationMetrics
from models.unet import UNet, UNet2
from models.init_weights import init_weights_light, weights_init
from datasets.transforms import *
from datasets.brats21 import BRATS21_2D
from evaluate.lossFunc import *
from evaluate.metrics import *

os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
torch.autograd.set_detect_anomaly(True)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 42
scaler = GradScaler()  # 混合精度训练
MetricsGo = EvaluationMetrics()  # 实例化评估指标类

def initialize_model(args):
    """初始化模型"""
    if args.model == 'unet':
        model = UNet(in_channels=args.in_channel, out_channels=args.out_channel)
    weights_init(model)
    
    model.to(DEVICE)
    return model

def load_data(args):
    """加载数据集"""
    TransMethods_train = Compose([
        ToTensor(),
        CenterRandomCrop(crop_size=(128, 128)),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        RandomRotation((0, 360)),
        Normalize()
    ])

    TransMethods_val = Compose([
        ToTensor(),
        Normalize()
    ])

    train_dataset = BRATS21_2D(
        txt_file=args.train_txt,
        transform=TransMethods_train,
        local_train=args.local_train,
        length=args.train_length,
    )

    val_dataset = BRATS21_2D(
        txt_file=args.val_txt,
        transform=TransMethods_val,
        local_train=args.local_train,
        length=args.val_length,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )
    # args.setattr('train_length', len(train_loader))
    # args.setattr('val_length', len(val_loader))
    
    print(f"已加载数据集, 训练集: {len(train_loader)}, 验证集: {len(val_loader)}")

    return train_loader, val_loader

def log_params(params, logs_path):
    """记录训练参数"""
    params_dict = {'Parameter': [str(p[0]) for p in list(params.items())],
                   'Value': [str(p[1]) for p in list(params.items())]}
    params_header = ["Parameter", "Value"]
    print(tabulate(params_dict, headers=params_header, tablefmt="grid"))
    custom_logger('='*40 + '\n' + "训练参数" +'\n' + '='*40 +'\n', logs_path, log_time=True)
    custom_logger(tabulate(params_dict, headers=params_header, tablefmt="grid"), logs_path)

def main(args):
    start_epoch = 0
    best_val_loss = float('inf')
    resume_tb_path = None

    """------------------------------------- 定义或获取路径 --------------------------------------------"""
    if args.resume:
        resume_path = args.resume
        print(f"Resuming training from {resume_path}")
        results_dir = os.path.join(*resume_path.split('/')[:-2])
        resume_tb_path = os.path.join(results_dir, 'tensorBoard')
        logs_dir = os.path.join(results_dir, 'logs')
        logs_file_name = [file for file in os.listdir(logs_dir) if file.endswith('.log')]
        logs_path = os.path.join(logs_dir, logs_file_name[0])
    else:
        os.makedirs(args.results_root, exist_ok=True)
        results_dir = os.path.join(args.results_root, get_current_date())
        results_dir = create_folder(results_dir)
        logs_dir = os.path.join(results_dir, 'logs')
        logs_path = os.path.join(logs_dir, f'{get_current_date()}.log')
        os.makedirs(logs_dir, exist_ok=True)

    """------------------------------------- 记录当前实验内容 --------------------------------------------"""
    exp_commit = args.commit if args.commit else input("请输入本次实验的更改内容: ")
    write_commit_file(os.path.join(results_dir, 'commits.md'), exp_commit)

    """------------------------------------- 模型实例化、初始化 --------------------------------------------"""
    model = initialize_model(args)
    total_params = sum(p.numel() for p in model.parameters())
    total_params = f'{total_params/1024**2:.2f} M'
    print(f"Total number of parameters: {total_params}")
    setattr(args, 'total_parms', total_params)

    """------------------------------------- 断点续传 --------------------------------------------"""
    if args.resume:
        print(f"Resuming training from checkpoint {args.resume}")
        checkpoint = torch.load(args.resume)
        best_val_loss = checkpoint['best_val_loss']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint {args.resume}")
        print(f"Best val loss: {best_val_loss:.4f} ✈ epoch {start_epoch}")
        cutoff_tb_data(resume_tb_path, start_epoch)
        print(f"Refix resume tb data step {resume_tb_path} up to step {start_epoch}")

    """------------------------------------- 载入数据集 --------------------------------------------"""
    train_loader, val_loader = load_data(args)

    """------------------------------------- 优化器 --------------------------------------------"""
    optimizer = AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.wd)

    """------------------------------------- 调度器 --------------------------------------------"""
    scheduler = CosineAnnealingLR(optimizer, T_max=args.cosine_T_max, eta_min=float(args.cosine_min_lr))

    """------------------------------------- 损失函数 --------------------------------------------"""
    loss_function = DiceLoss()
    # loss_function = nn.BCEWithLogitsLoss()

    """------------------------------------- 输出参数列表 --------------------------------------"""
    log_params(vars(args), logs_path)

    """------------------------------------- 训练模型 --------------------------------------------"""
    train(model, 
          Metrics=MetricsGo, 
          train_loader=train_loader,
          val_loader=val_loader, 
          scaler=scaler, 
          optimizer=optimizer,
          scheduler=scheduler,
          loss_function=loss_function,
          num_epochs=args.epochs, 
          device=DEVICE, 
          results_dir=results_dir,
          logs_path=logs_path,
          start_epoch=start_epoch,
          best_val_loss=best_val_loss,
          tb=args.tb,
          interval=args.interval,
          save_max=args.save_max,
          early_stopping_patience=args.early_stop_patience,
          resume_tb_path=resume_tb_path)

def parse_args_from_yaml(yaml_file):
    """从 YAML 文件中解析配置参数"""
    assert os.path.exists(yaml_file), FileNotFoundError(f"Config file not found at {yaml_file}")
    with open(yaml_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train args')
    parser.add_argument('--config', type=str, 
                        default='/root/workspace/SliceMedix/src/configs/default.yaml', 
                        help='Path to the configuration YAML file')
    parser.add_argument('--resume', type=str, 
                        default=None, 
                        help='Path to the checkpoint to resume training from')
    parser.add_argument('--resume_tb_path', type=str,
                        default=None, 
                        help='Path to the TensorBoard logs to resume from')
    args = parser.parse_args()
    args = argparse.Namespace(**parse_args_from_yaml(args.config))
    main(args=args)