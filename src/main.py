import os
import torch
import torch.nn as nn
import argparse
from train import train
from tabulate import tabulate

from torch.utils.data import DataLoader
from torch.optim import Adam, SGD, RMSprop, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.amp import GradScaler


# from nets.unet3d.unet3d import *
# from nets.unet3d.fusion_unet import *
# from nets.model_weights_init import init_weights_light
from evaluate.lossFunction import DiceLoss, CELoss, FocalLoss
from utils.get_commits import *
from readDatasets.BraTS import BraTS21_3d
from transforms import *
from utils.log_writer import *
from utils.split_dataList import dataSpliter
from utils.reload_tb_events import *
from metrics import EvaluationMetrics

os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
torch.autograd.set_detect_anomaly(True)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
scaler = GradScaler() # 混合精度训练
MetricsGo = EvaluationMetrics() # 实例化评估指标类

'''
 TODO:
    - [ ] 使用unet3d_dilation, 查看效果    | DDL: 2024//
    - [ ] 使用unet3d_bn_5x5, 查看效果    | DDL: 2024//
    - [ ] 使用unet3d_bn_res / dilation查看效果    | DDL: 2024//
    - [ ] 添加注意力机制， SE、CBAM等
'''

def main(args):
    start_epoch = 0
    best_val_loss = float('inf')
    resume_tb_path = None

    """------------------------------------- 定义或获取路径 --------------------------------------------"""
    # if args.resume: # 如果断点续传可用，则使用断点续训的路径  # TODO: 断点续传
    #     resume_path = args.resume
    #     print(f"Resuming training from {resume_path}")
    #     results_dir = ('/').join(resume_path.split('/')[:-2])
    #     resume_tb_path = os.path.join(results_dir, 'tensorBoard')
    #     logs_dir = os.path.join(results_dir, 'logs')
    #     logs_file_name = [file for file in os.listdir(logs_dir) if file.endswith('.log')]
    #     logs_path = os.path.join(logs_dir, logs_file_name[0])
    # else:
    #     # 创建结果保存路径
    #     os.makedirs(args.results_root, exist_ok=True)
    #     results_dir = os.path.join(args.results_root, get_current_date())
    #     results_dir = create_folder(results_dir) # 创建时间文件夹

    #     logs_dir = os.path.join(results_dir, 'logs')
    #     logs_path = os.path.join(logs_dir, f'{get_current_date()}.log')
    #     os.makedirs(logs_dir, exist_ok=True)

    """------------------------------------- 记录当前实验内容 --------------------------------------------"""
    # if args.commit == None:
    #     exp_commit = input("请输入本次实验的更改内容: ")
    # else:
    #     exp_commit = args.commit
    # # write_commit_file(os.path.join(results_dir,'commits.md'), exp_commit)

    """------------------------------------- 模型实例化、初始化 --------------------------------------------"""
    # if args.model == 'unet':
    
    # init_weights_light(model)
    # model.to(DEVICE)
    # 计算模型参数量
    # total_params = sum(p.numel() for p in model.parameters())
    # total_params = f'{total_params/1024**2:.2f} M'
    # print(f"Total number of parameters: {total_params}")
    # setattr(args, 'total_parms', total_params)
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
    
    # 打印网络结构

    summary(model, input_size=(1, 4, 128, 128, 128))

    """------------------------------------- 获取数据列表csv --------------------------------------------"""
    train_csv = os.path.join(args.data_root, "train.csv")
    val_csv = os.path.join(args.data_root, "val.csv")
    test_csv = os.path.join(args.data_root, "test.csv")
    root = args.data_root
    path_data = os.path.join(root, "BraTS2021_Training_Data")
    assert os.path.exists(root), f"{root} not exists."
    
    """------------------------------------- 划分数据集 --------------------------------------------"""
    if args.data_split:
        dataspliter =  dataSpliter(path_data, train_split=args.ts, val_split=args.vs, seed=RANDOM_SEED)
        train_list, test_list, val_list = dataspliter.data_split()
        dataspliter.save_as_csv(train_list, train_csv)
        dataspliter.save_as_csv(test_list, val_csv)
        dataspliter.save_as_csv(val_list, test_csv)
    else:
        delattr(args, 'ts')
        delattr(args, 'vs')

    """------------------------------------- 载入数据集 --------------------------------------------"""
    TransMethods_train = data_transform(transform=Compose([RandomCrop3D(size=args.trainCropSize),    # 随机裁剪
                                                        tioRandomFlip3d(),                 # 随机翻转
                                                        Normalize(mean=(0.114, 0.090, 0.170, 0.096), std=(0.199, 0.151, 0.282, 0.174)),   # 标准化
                                                        # tioRandomElasticDeformation3d(),
                                                        # tioRandomAffine(),          # 随机旋转
                                                        # tioZNormalization(),               # 归一化
                                      ])) #! 不加噪声 不加噪声 不加噪声
    
    TransMethods_val = data_transform(transform=Compose([RandomCrop3D(size=args.valCropSize),    # 随机裁剪
                                                         tioRandomFlip3d(),   
                                                         Normalize(mean=(0.114, 0.090, 0.170, 0.096), std=(0.199, 0.151, 0.282, 0.174)),   # 标准化
                                                        #  tioZNormalization(),               # 归一化
                                      ]))
    
    assert args.data_scale in ['debug', 'small', 'full'], "data_scale must be 'debug', 'small' or 'full'!"
    if args.data_scale == 'small':
        # 载入部分数据集
        setattr(args, 'trainSet_len', 480)
        setattr(args, 'valSet_len', 60)
        train_dataset = BraTS21_3d(train_csv, 
                                   transform=TransMethods_train,
                                   local_train=True, 
                                   length=args.trainSet_len)
        
        val_dataset   = BraTS21_3d(val_csv, 
                                   transform=TransMethods_val, 
                                   local_train=True, 
                                   length=args.valSet_len)
    elif args.data_scale == 'debug':
        # 载入部分数据集
        setattr(args, 'trainSet_len', 80)
        setattr(args, 'valSet_len', 10)
        train_dataset = BraTS21_3d(train_csv, 
                                   transform=TransMethods_train,
                                   local_train=True, 
                                   length=args.trainSet_len)

        val_dataset   = BraTS21_3d(val_csv, 
                                   transform=TransMethods_val, 
                                   local_train=True, 
                                   length=args.valSet_len)
    else:       # 载入全部数据集
        train_dataset = BraTS21_3d(train_csv, 
                                transform=TransMethods_train)
        
        val_dataset   = BraTS21_3d(val_csv, 
                                transform=TransMethods_val)
        setattr(args, 'trainSet_len', len(train_dataset))
        setattr(args, 'valSet_len', len(val_dataset))

    assert args.nw > 0 and args.nw <= 8 , "num_workers must be in (0, 8]!"
    train_loader = DataLoader(train_dataset, 
                              batch_size=args.bs, 
                              num_workers=args.nw,
                              shuffle=True)
    
    val_loader   = DataLoader(val_dataset, 
                              batch_size=args.bs, 
                              num_workers=args.nw,
                              shuffle=False)

    """------------------------------------- 优化器 --------------------------------------------"""
    assert args.optimizer in ['AdamW', 'SGD', 'RMSprop'], \
        f"optimizer must be 'AdamW', 'SGD' or 'RMSprop', but got {args.optimizer}."
    if args.optimizer == 'AdamW':
        optimizer = AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.wd) # 会出现梯度爆炸或消失
    elif args.optimizer == 'SGD':
        optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9)
    elif args.optimizer == 'RMSprop':
        optimizer = RMSprop(model.parameters(), lr=args.lr, alpha=0.9, eps=1e-8)
    else:
        raise ValueError("optimizer must be 'AdamW', 'SGD' or 'RMSprop'.")
    

    """------------------------------------- 调度器 --------------------------------------------"""
    if args.scheduler == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=args.reduce_factor, patience=args.reduce_patience)
        delattr(args,'cosine_T_max')
        delattr(args,'cosine_min_lr')
    elif args.scheduler == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.cosine_T_max, eta_min=args.cosine_min_lr)
        delattr(args, 'reduce_patience')
        delattr(args, 'reduce_factor')
    else:
        scheduler = None
        delattr(args,'reduce_patience')
        delattr(args,'reduce_factor')
        delattr(args,'cosine_T_max')
        delattr(args,'cosine_min_lr')
    
    
    """------------------------------------- 损失函数 --------------------------------------------"""
    assert args.loss in ['DiceLoss', 'CELoss', 'FocalLoss'], \
        f"loss must be 'DiceLoss' or 'CELoss' or 'FocalLoss', but got {args.loss}."
    if args.loss == 'CELoss':
        loss_function = CELoss()
    elif args.loss == 'FocalLoss':
        loss_function = FocalLoss()
    else:
        loss_function = DiceLoss()
    
    """--------------------------------------- 输出参数列表 --------------------------------------"""
    # 将参数转换成字典,并输出参数列表
    params = vars(args) 
    params_dict = {}
    params_dict['Parameter']=[str(p[0]) for p in list(params.items())]
    params_dict['Value']=[str(p[1]) for p in list(params.items())]
    params_header = ["Parameter", "Value"]

    # 标准输出
    print(tabulate(params_dict, headers=params_header, tablefmt="grid"))
    # 重定向输出
    custom_logger('='*40 + '\n' + "训练参数" +'\n' + '='*40 +'\n', logs_path, log_time=True)
    custom_logger(tabulate(params_dict, headers=params_header, tablefmt="grid"), logs_path)
    
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

if __name__ == "__main__":
    # 实例化参数容器
    parser = argparse.ArgumentParser(description="Train args")

    # 训练基本设置
    parser.add_argument("--epochs",                         type=int, 
                        default=10,                         help="num_epochs")
    parser.add_argument("--nw",                             type=int, 
                        default=4,                          help="num_workers")
    parser.add_argument("--bs",                             type=int, 
                        default=1,                          help="batch_size")
    
    # 模型相关的参数
    parser.add_argument("--model",                          type=str, 
                        default="fm_unet3d_dilation_fusion", help="")
    parser.add_argument("--fusion_flag",                    type=str, 
                        default=True,                       help="fusion method")
    parser.add_argument("--input_channels",                 type=int, 
                        default=4,                          help="input channels")
    parser.add_argument("--output_channels",                type=int, 
                        default=4,                          help="output channels")
    parser.add_argument("--total_parms",                    type=int, 
                        default=None,  required=False,      help="total parameters")
    parser.add_argument("--early_stop_patience",            type=int, 
                        default=0, help="early stop patience")
    parser.add_argument("--resume",                         type=str, 
                        default=None,                       help="resume training from checkpoint")
    
    # 训练参数
    parser.add_argument("--loss",                           type=str, 
                        default="DiceLoss",                 help="loss function: ['DiceLoss', 'CELoss', 'FocalLoss']")
    parser.add_argument("--loss_type",                      type=str, 
                        default="subarea_mean",             help="loss type to grad")
    parser.add_argument("--save_max",                       type=int, 
                        default=5,                          help="ckpt max save number")
    parser.add_argument("--interval",                       type=int, 
                        default=1,                          help="checkpoint interval")
    
    # 优化器
    parser.add_argument("--optimizer",                      type=str, 
                        default="AdamW",                    help="optimizers: ['AdamW', 'SGD', 'RMSprop']")
    parser.add_argument("--lr",                             type=float, 
                        default=3e-4,                       help="learning rate")
    parser.add_argument("--wd",                             type=float, 
                        default=1e-5,                       help="weight decay")
    
    # 学习率调度器
    parser.add_argument("--scheduler",                      type=str, 
                        default='CosineAnnealingLR',        help="schedulers:['ReduceLROnPlateau', 'CosineAnnealingLR']")
    parser.add_argument("--cosine_min_lr",                  type=float, 
                        default=1e-8,                       help="CosineAnnealingLR min lr")
    parser.add_argument("--cosine_T_max",                   type=int, 
                        default=300,                        help="CosineAnnealingLR T max")

    parser.add_argument("--reduce_patience",                type=int, 
                        default=3,                          help="ReduceLROnPlateau scheduler patience")
    parser.add_argument("--reduce_factor",                  type=float, 
                        default=0.9,                        help="ReduceLROnPlateau scheduler factor")
    
    # 数据参数
    parser.add_argument("--tb",                             type=bool, 
                        default=True,                       help="TensorBoard True or False")
    parser.add_argument("--data_split",                     type=bool, 
                        default=False,                      help="data split True or False")
    parser.add_argument("--ts",                             type=float, 
                        default=0.8,                        help="train_split_rata")
    parser.add_argument("--vs",                             type=float, 
                        default=0.1,                        help="val_split_rate")
    parser.add_argument("--data_root" ,                     type=str, 
                        default="./brats21_local",          help="data root")
    parser.add_argument("--data_scale",                     type=str, 
                        default="debug",                    help="loading data scale")
    parser.add_argument("--trainSet_len",                   type=int, 
                        default=100,                        help="train length")
    parser.add_argument("--valSet_len",                     type=int, 
                        default=12,                         help="val length")
    parser.add_argument("--trainCropSize",                  type=lambda x: tuple(map(int, x.split(','))), 
                        default=(128, 128, 128),            help="crop size")
    parser.add_argument("--valCropSize", type=lambda x: tuple(map(int, x.split(','))), default=(128, 128, 128),
                        help="crop size")
    
    parser.add_argument("--results_root",                   type=str, 
                        default="./results",                help="result path")
    
    parser.add_argument("--commit",                         type=str, 
                        default='debug',                    help="training commit")

    args = parser.parse_args()
    main(args=args)