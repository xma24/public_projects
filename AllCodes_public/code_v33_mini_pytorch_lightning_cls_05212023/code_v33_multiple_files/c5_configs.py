import os

import numpy as np
import torch


class Configs:

    """+++ experiment setting."""

    logger_root = "./work_dirs/main_logger/"
    work_dirs = "./work_dirs/"
    data_root = "./data/"
    random_seed = 42
    workers = 16
    pin_memory = True
    dataset_name = "cifar10"
    data_dir = os.path.join(data_root, dataset_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cpus = False
    num_gpus = "autocount"
    num_nodes = 1

    """ +++ data configs."""
    num_classes = 10
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    classes = None
    if classes is None:
        classes = np.arange(num_classes)

    """ +++ model configs"""
    backbone_name = "efficientnet"
    pretrained_weights = "/home/xma24/vscode/public_projects/public_projects/AllCodes_public/code_v33_mini_pytorch_lightning_cls_05212023/code_v33_multiple_files/pt-cifar10-efficientnet/2et5p6oe/checkpoints/epoch=48-val_loss=0.40-user_metric=0.87.ckpt"
    pretrained_weights_max_epoch = 1
    pre_lr = 0.00001

    """ +++ training configs."""
    training_mode = True
    batch_size = 32 * 8
    lr = 0.08
    backbone_lr = 0.01
    T_max = 100  # for cosineAnn; The same with max epoch
    eta_min = 1e-6  # for cosineAnn
    T_0 = 5  # for cosineAnnWarm; cosineAnnWarmDecay
    T_mult = 9  # for cosineAnnWarm; cosineAnnWarmDecay
    max_epochs = T_0 + T_mult * T_0
    print_interval = 20
    scheduler = "cosineAnn"
    opt = "SGD"
    step_ratio = 0.3  # for StepLR
    gamma = 0.1  # for StepLR
    single_lr = False
    precision = 16
    strategy = "ddp"
    accelerator = "gpu"
    progress_bar_refresh_rate = 1
    logger_name = "wandb"  # "neptune", "csv", "wandb"
    wandb_name = "pt-" + dataset_name + "-" + backbone_name

    """ +++ validation configs."""
    val_interval = 1
    val_batch_size = 16
