import os

import numpy as np
import torch
from umap.umap_ import find_ab_params


class Configs:
    ### Data Config:
    logger_root = "./work_dirs/main_logger/"
    data_root = "/data/SSD1/data/"
    work_dirs = "./work_dirs/"
    random_seed = 42
    workers = 16
    pin_memory = True

    dataset_name = "umap_cifar10"
    num_classes = 10

    # dataset_name = "imagenet"
    # num_classes = 1000

    data_dir = os.path.join(data_root, dataset_name)

    if dataset_name == "imagenet":
        assert num_classes == 1000, print(f"class name and class number not matched.")
        assert data_root == "/data/SSD1/data/", print(
            f"class name and data root folder not matched."
        )
    elif dataset_name == "cifar10":
        assert num_classes == 10, print(f"class name and class number not matched.")
        assert data_root == "/data/SSD1/data/", print(
            f"class name and data root folder not matched."
        )

    classes = None

    image_raw_size = 256
    image_crop_size = 224

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if classes is None:
        classes = np.arange(num_classes)

    ### Net Config:
    backbone_name = "vit"
    lr = 0.008
    opt = "SGD"
    backbone_lr = 0.001

    ### Training Config:
    subtrain = False
    subtrain_ratio = 0.1

    batch_size = 16 * 8 * 4
    # batch_size = Utils.get_max_batchsize(data_dir)

    # num_epochs = 10
    print_interval = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scheduler = "cosineAnn"

    step_ratio = 0.3  # for StepLR
    gamma = 0.1  # for StepLR

    pretrained_weights = None
    pretrained_weights_max_epoch = 40
    pre_lr = 0.01

    single_lr = False

    logger_name = "wandb"  # "neptune", "csv", "wandb"
    cpus = False

    wandb_name = "pt-" + dataset_name + "-" + backbone_name

    training_mode = True
    num_gpus = "autocount"
    num_nodes = 1
    precision = 16
    strategy = "ddp"
    accelerator = "gpu"
    progress_bar_refresh_rate = 1

    T_max = 100  # for cosineAnn; The same with max epoch
    eta_min = 1e-6  # for cosineAnn
    T_0 = 5  # for cosineAnnWarm; cosineAnnWarmDecay
    T_mult = 19  # for cosineAnnWarm; cosineAnnWarmDecay
    # max_epochs = T_0 + T_mult * T_0
    max_epochs = 1

    ### Validation Config:
    subval = False
    subval_ratio = 0.1
    val_interval = 1
    val_batch_size = 16
    val_interval_batches = 200

    """ +++ umap settings."""
    umap_n_components = 2
    match_nonparametric_umap = False
    umap_lr = 0.001
    umap_beta = 1.0
    umap_min_dist = 0.1
    # umap_reconstruction_loss = 0.5
    umap_n_neighbors = 15
    umap_metric = "euclidean"
    umap_random_state = 42
    umap_batch_size = 16
    umap_num_workers = 16
    expr_index = "2"
    dim = umap_n_components
    negative_sample_rate = 5
    _a, _b = find_ab_params(1.0, umap_min_dist)
    negative_force = 1.0
    continuous = True
