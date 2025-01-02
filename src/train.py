# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2024/12/30 14:53:59
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: 训练流程
*      VERSION: v1.0
=================================================
'''


import os
import time
import torch
import numpy as np

from datetime import datetime
from tabulate import tabulate

# 日志记录
import logging
from utils.logger_tools import *
from utils.shell_tools import *
from torch.utils.tensorboard import SummaryWriter 

# 数据集处理
from datasets.transforms import *
from datasets.brats21 import BRATS21_2D

# 计算
from evaluate.lossFunc import *
from evaluate.metrics import *

# 模型
# from models.unet import UNet2
from utils.ckpt_tools import *

# 训练
from train_and_val import *



# constant
TB_PORT = 6007
RANDOM_SEED = 42
scheduler_start_epoch = 0
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)                 #让显卡产生的随机数一致

logger = logging.getLogger(__name__)


get_current_date = lambda: datetime.now().strftime('%Y-%m-%d')
get_current_time = lambda: datetime.now().strftime('%H-%M-%S')


def log_metrics(epoch, model_name, optimizer_name, scheduler_name, loss_func_name, current_lr, 
                val_cost_time, val_scores, val_running_loss, val_et_loss, val_tc_loss, val_wt_loss, logs_path):
    """记录验证指标"""
    metric_table_header = ["Metric_Name", "MEAN", "ET", "TC", "WT"]
    metric_table_left = ["Dice", "Jaccard", "Accuracy", "Precision", "Recall", "F1", "F2"]

    val_info_str =  f"=== [Epoch {epoch}] ===\n"\
                    f"- Model:    {model_name}\n"\
                    f"- Optimizer:{optimizer_name}\n"\
                    f"- Scheduler:{scheduler_name}\n"\
                    f"- LossFunc: {loss_func_name}\n"\
                    f"- Lr:       {current_lr:.6f}\n"\
                    f"- val_cost_time:{val_cost_time:.4f}s ⏱️\n"

    metric_scores_mapping = {metric: val_scores[f"{metric}_scores"] for metric in metric_table_left}
    metric_table = [[metric,
                    f"{metric_scores_mapping[metric][0]:.4f}",
                    f"{metric_scores_mapping[metric][1]:.4f}",
                    f"{metric_scores_mapping[metric][2]:.4f}",
                    f"{metric_scores_mapping[metric][3]:.4f}"] for metric in metric_table_left]
    loss_str = f"Mean Loss: {val_running_loss:.4f}, ET: {val_et_loss:.4f}, TC: {val_tc_loss:.4f}, WT: {val_wt_loss:.4f}\n"
    table_str = tabulate(metric_table, headers=metric_table_header, tablefmt='grid')
    metrics_info = val_info_str + table_str + '\n' + loss_str  
    
    custom_logger(metrics_info, logs_path)
    print(metrics_info)

def save_best_model(model, optimizer, scaler, best_epoch, best_val_loss, best_dice, ckpt_dir, model_name, loss_func_name, date_time_str, save_counter, save_max):
    """保存最佳模型"""
    best_ckpt_path = os.path.join(ckpt_dir, f'best@e{best_epoch}_{model_name}__{loss_func_name.lower()}{best_val_loss:.4f}_dice{best_dice:.4f}_{date_time_str}_{save_counter}.pth')
    if save_counter > save_max:
        removed_ckpt = [ckpt for ckpt in os.listdir(ckpt_dir) if (ckpt.endswith('.pth') and (int(ckpt.split('.')[-2].split('_')[-1]) == int(save_counter - save_max)))]
        os.remove(os.path.join(ckpt_dir, removed_ckpt[0]))
        print(f"🗑️ Due to reach the max save amount, Removed {removed_ckpt[0]}")
    save_checkpoint(model, optimizer, scaler, best_epoch, best_val_loss, best_ckpt_path)

def train(model, Metrics, train_loader, val_loader, scaler, optimizer, scheduler, loss_function, 
          num_epochs, device, results_dir, logs_path, start_epoch, best_val_loss, 
          tb=False,  
          interval=10, 
          save_max=10, 
          early_stopping_patience=10,
          resume_tb_path=None):
    
    best_epoch = 0
    save_counter = 0
    early_stopping_counter = 0
    date_time_str = get_current_date() + '_' + get_current_time()
    end_epoch = start_epoch + num_epochs
    model_name = model.__class__.__name__
    optimizer_name = optimizer.__class__.__name__
    scheduler_name = scheduler.__class__.__name__
    loss_func_name = loss_function.__class__.__name__

    if resume_tb_path:
        tb_dir = resume_tb_path
    else:
        tb_dir = os.path.join(results_dir, f'tensorBoard')
    ckpt_dir = os.path.join(results_dir, f'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    if scheduler:
        current_lr = scheduler.get_last_lr()[0]
    else:
        current_lr = optimizer.param_groups[0]["lr"]

    logger.info(f'开始训练, 训练轮数:{num_epochs}, {model_name}模型写入tensorBoard, 使用 {optimizer_name} 优化器, 学习率: {current_lr}, 损失函数: {loss_func_name}')

    writer = SummaryWriter(tb_dir)
    
    print(f'{model_name}模型写入tensorBoard, 使用 {optimizer_name} 优化器, 学习率: {optimizer.param_groups[0]["lr"]}, 损失函数: {loss_func_name}')
    for epoch in range(start_epoch, end_epoch):
        epoch += 1
        if scheduler:
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = optimizer.param_groups[0]["lr"]
        
        """-------------------------------------- 训练过程 --------------------------------------------------"""
        print(f"=== Training on [Epoch {epoch}/{end_epoch}] ===:")
        
        train_mean_loss = 0.0
        start_time = time.time()
        train_running_loss, train_et_loss, train_tc_loss, train_wt_loss = train_one_epoch(model, train_loader, optimizer, loss_function, scaler, device)

        
        if scheduler_name == 'CosineAnnealingLR' and epoch > scheduler_start_epoch:
            scheduler.step()
        writer.add_scalars(f'{loss_func_name}/train',
                           {'Mean':train_mean_loss, 'ET': train_wt_loss, 'TC': train_tc_loss, 'WT': train_wt_loss}, epoch)
        end_time = time.time()
        train_cost_time = end_time - start_time
        print(f"- Train mean loss: {train_running_loss:.4f}\n"
              f"- ET loss: {train_et_loss:.4f}\n"
              f"- TC loss: {train_tc_loss:.4f}\n"
              f"- WT loss: {train_wt_loss:.4f}\n"
              f"- Cost time: {train_cost_time/60:.2f}mins ⏱️\n")
        
        """-------------------------------------- 验证过程 --------------------------------------------------"""
        if (epoch) % interval == 0:
            print(f"=== Validating on [Epoch {epoch}/{end_epoch}] ===:")
            
            start_time = time.time()
            val_running_loss, val_et_loss, val_tc_loss, val_wt_loss, Metrics_list= validate_one_epoch(model, Metrics, val_loader, loss_function, device)
            
            val_scores = {
                'epoch': epoch,
                'Dice_scores': Metrics_list[0],
                'Jaccard_scores': Metrics_list[1],
                'Accuracy_scores': Metrics_list[2],
                'Precision_scores': Metrics_list[3],
                'Recall_scores': Metrics_list[4],
                'F1_scores': Metrics_list[5],
                'F2_scores': Metrics_list[6]
            }
            
            writer.add_scalars(f'{loss_func_name}/val', 
                               {'Mean':val_running_loss, 'ET': val_et_loss, 'TC': val_tc_loss, 'WT': val_wt_loss}, 
                               epoch)
            writer.add_scalars(f'{loss_func_name}/Mean', 
                               {'Train':train_mean_loss, 'Val':val_running_loss}, 
                               epoch)
            writer.add_scalars(f'{loss_func_name}/ET',
                               {'Train':train_et_loss, 'Val':val_et_loss}, 
                               epoch)
            writer.add_scalars(f'{loss_func_name}/TC',
                               {'Train':train_tc_loss, 'Val':val_tc_loss}, 
                               epoch)
            writer.add_scalars(f'{loss_func_name}/WT',
                               {'Train':train_wt_loss, 'Val':val_wt_loss}, 
                               epoch)

            if epoch == start_epoch+1:
                start_tensorboard(tb_dir, port=TB_PORT)

            if tb: 
                writer.add_scalars('metrics/Dice_coeff',
                                    {
                                        'Mean':val_scores['Dice_scores'][0], 
                                        'ET': val_scores['Dice_scores'][1], 
                                        'TC': val_scores['Dice_scores'][2], 
                                        'WT': val_scores['Dice_scores'][3]
                                    },
                                    epoch)

                writer.add_scalars('metrics/Jaccard_index', 
                                   {
                                       'Mean':val_scores['Jaccard_scores'][0],
                                       'ET': val_scores['Jaccard_scores'][1], 
                                       'TC': val_scores['Jaccard_scores'][2],
                                       'WT': val_scores['Jaccard_scores'][3]
                                    },
                                    epoch)   

                writer.add_scalars('metrics/Accuracy',
                                    {
                                        'Mean':val_scores['Accuracy_scores'][0],
                                        'ET': val_scores['Accuracy_scores'][1],
                                        'TC': val_scores['Accuracy_scores'][2],
                                        'WT': val_scores['Accuracy_scores'][3]
                                    },
                                    epoch)
                
                writer.add_scalars('metrics/Precision', 
                                    {
                                        'Mean':val_scores['Precision_scores'][0],
                                        'ET': val_scores['Precision_scores'][1],
                                        'TC': val_scores['Precision_scores'][2],
                                        'WT': val_scores['Precision_scores'][3]
                                    },
                                    epoch)
                
                writer.add_scalars('metrics/Recall', 
                                    {
                                        'Mean':val_scores['Recall_scores'][0],
                                        'ET': val_scores['Recall_scores'][1],
                                        'TC': val_scores['Recall_scores'][2],
                                        'WT': val_scores['Recall_scores'][3]
                                    },
                                    epoch)
                
                writer.add_scalars('metrics/F1', 
                                    {
                                        'Mean':val_scores['F1_scores'][0],
                                        'ET': val_scores['F1_scores'][1],
                                        'TC': val_scores['F1_scores'][2],
                                        'WT': val_scores['F1_scores'][3]
                                    },
                                    epoch) 
                writer.add_scalars('metrics/F2', 
                                    {
                                        'Mean':val_scores['F2_scores'][0],
                                        'ET': val_scores['F2_scores'][1],
                                        'TC': val_scores['F2_scores'][2],
                                        'WT': val_scores['F2_scores'][3]
                                    },
                                    epoch)                               
            
            end_time = time.time()
            val_cost_time = end_time - start_time
            
            log_metrics(epoch, model_name, optimizer_name, scheduler_name, loss_func_name, 
                        current_lr, val_cost_time, val_scores, val_running_loss, val_et_loss, val_tc_loss, val_wt_loss, logs_path)
            
            """------------------------------------- 保存权重文件 --------------------------------------------"""
            best_dice = val_scores['Dice_scores'][0]
            if val_running_loss < best_val_loss:
                early_stopping_counter = 0
                best_val_loss = val_running_loss
                best_epoch = epoch
                with open(os.path.join(os.path.dirname(logs_path), "current_log.txt"), 'a') as f:
                    f.write(f"=== Best EPOCH {best_epoch} ===:\n"\
                            f"@ {get_current_date() + ' ' + get_current_time()}\n"\
                            f"current lr : {current_lr:.6f}\n"\
                            f"loss: Mean:{val_running_loss:.4f}\t ET: {val_et_loss:.4f}\t TC: {val_tc_loss:.4f}\t WT: {val_wt_loss:.4f}\n"
                            f"mean dice : {val_scores['Dice_scores'][0]:.4f}\t" \
                            f"ET : {val_scores['Dice_scores'][1]:.4f}\t"\
                            f"TC : {val_scores['Dice_scores'][2]:.4f}\t" \
                            f"WT : {val_scores['Dice_scores'][3]:.4f}\n\n")
                    
                save_counter += 1
                save_best_model(model, optimizer, scaler, best_epoch, best_val_loss, best_dice, ckpt_dir, model_name, loss_func_name, date_time_str, save_counter, save_max)
            else:
                if early_stopping_counter == 0 :
                    continue
                else:
                    early_stopping_counter += 1
                    if early_stopping_counter >= early_stopping_patience:
                        print(f"🎃 Early stopping at epoch {epoch} due to no improvement in validation loss.")
                        break
                
    print(f"😃😃😃Train finished. Best val loss: 👉{best_val_loss:.4f} at epoch {best_epoch}")
    writer.close()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train args')
    parser.add_argument('--train_txt', type=str, required=True, help='Path to the training data txt file')
    parser.add_argument('--val_txt', type=str, required=True, help='Path to the validation data txt file')
    parser.add_argument('--test_txt', type=str, required=True, help='Path to the test data txt file')
    args = parser.parse_args()

    trans_train = Compose([
        ToTensor(),
        RandomCrop(crop_size=(196, 196)),
        RandomHorizontalFlip(flip_prob=0.5),
        RandomVerticalFlip(flip_prob=0.5),
        RandomRotation((0, 360)),
        Normalize(),
    ])

    trans_val = Compose([
        ToTensor(),
        Normalize(),
    ])

    train_datasets = BRATS21_2D(args.train_txt, transform=trans_train)
    val_datasets = BRATS21_2D(args.val_txt, transform=trans_val)
    test_datasets = BRATS21_2D(args.test_txt, transform=trans_val)

    print(f"训练数据：{len(train_datasets)}, \n",
          f"验证数据：{len(val_datasets)}\n", 
          f"测试数据: {len(test_datasets)}"
    )

    print(train_datasets[0][0].shape, train_datasets[0][0].dtype)
    print(train_datasets[0][1].shape, train_datasets[0][1].dtype)

    model = UNet(4, 32, 4)
    print(model)