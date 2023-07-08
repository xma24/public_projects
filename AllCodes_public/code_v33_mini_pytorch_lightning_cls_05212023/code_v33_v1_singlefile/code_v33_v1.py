import datetime
import logging
import os
import sys
import time

import numpy as np
import pytorch_lightning as pl
import pytz
import timm
import torch
import torch.distributed as dist
import torch.nn as nn
import wandb
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.strategies import BaguaStrategy, DDPStrategy
from pytorch_lightning.strategies.deepspeed import DeepSpeedStrategy
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR


class TorchModel:
    @staticmethod
    def get_model(num_classes):
        import timm

        keyworlds = "resnet50"
        selected_listed_models = TorchModel.get_timm_model_list(keyworlds)
        print(f"==>> selected_listed_models: {selected_listed_models}")

        model_name = "efficientnet_b0.ra_in1k"
        init_model = timm.create_model(model_name, pretrained=True)
        feature_dim = init_model.get_classifier().in_features

        backbone = nn.Sequential()
        init_classifier = nn.Sequential()
        named_modules_list = list(init_model.named_children())
        # module_name_list = [name for name, _ in named_modules_list]

        for i, (name, module) in enumerate(init_model.named_children()):
            if i >= len(named_modules_list) - 1:
                init_classifier.add_module(name, module)
                # break
            else:
                backbone.add_module(name, module)

        classifier = nn.Linear(feature_dim, num_classes)

        return init_model, backbone, init_classifier, classifier

    @staticmethod
    def get_timm_model_list(keyworlds):
        listed_models = timm.list_models("*")
        # print("==>> listed_models: ", listed_models)
        print("==>> listed_models len: ", len(listed_models))

        listed_pretraiend_models = timm.list_models(pretrained=True)
        # print("==>> listed_pretraiend_models: ", listed_pretraiend_models)
        print("==>> listed_pretraiend_models len: ", len(listed_pretraiend_models))

        selected_listed_models = timm.list_models("*" + keyworlds + "*")
        print("==>> selected_listed_models: ", selected_listed_models)
        print("==>> selected_listed_models len: ", len(selected_listed_models))

        return selected_listed_models

    @staticmethod
    def check_weights_loading(to_test_model):
        import urllib

        url, filename = (
            "https://github.com/pytorch/hub/raw/master/images/dog.jpg",
            "./work_dirs/dog.jpg",
        )
        try:
            urllib.URLopener().retrieve(url, filename)
        except:
            urllib.request.urlretrieve(url, filename)

        url_imagenet_classes, filename_imagenet_classes = (
            "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt",
            "./work_dirs/imagenet_classes.txt",
        )
        try:
            urllib.URLopener().retrieve(url_imagenet_classes, filename_imagenet_classes)
        except:
            urllib.request.urlretrieve(url_imagenet_classes, filename_imagenet_classes)

        from PIL import Image
        from torchvision import transforms

        input_image = Image.open(filename)
        preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(
            0
        )  # create a mini-batch as expected by the model

        if torch.cuda.is_available():
            input_batch = input_batch.to(TrainingConfig.device)
            to_test_model.to(TrainingConfig.device)

        to_test_model.eval()
        with torch.no_grad():
            output = to_test_model(input_batch)

        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        with open(filename_imagenet_classes, "r") as f:
            categories = [s.strip() for s in f.readlines()]

        print(f"\n prediction: \n")
        top5_prob, top5_catid = torch.topk(probabilities, 5)
        for i in range(top5_prob.size(0)):
            print(categories[top5_catid[i]], top5_prob[i].item())


class Utils:
    @staticmethod
    def get_logger(name, log_file=None, log_level=logging.INFO, file_mode="w"):
        logger_initialized = {}

        logger = logging.getLogger(name)
        if name in logger_initialized:
            return logger
        # handle hierarchical names
        # e.g., logger "a" is initialized, then logger "a.b" will skip the
        # initialization since it is a child of "a".
        for logger_name in logger_initialized:
            if name.startswith(logger_name):
                return logger

        # handle duplicate logs to the console
        # Starting in 1.8.0, PyTorch DDP attaches a StreamHandler <stderr> (NOTSET)
        # to the root logger. As logger.propagate is True by default, this root
        # level handler causes logging messages from rank>0 processes to
        # unexpectedly show up on the console, creating much unwanted clutter.
        # To fix this issue, we set the root logger's StreamHandler, if any, to log
        # at the ERROR level.
        for handler in logger.root.handlers:
            if type(handler) is logging.StreamHandler:
                handler.setLevel(logging.ERROR)

        stream_handler = logging.StreamHandler()
        handlers = [stream_handler]

        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
        else:
            rank = 0

        # only rank 0 will add a FileHandler
        if rank == 0 and log_file is not None:
            # Here, the default behaviour of the official logger is 'a'. Thus, we
            # provide an interface to change the file mode to the default
            # behaviour.
            file_handler = logging.FileHandler(log_file, file_mode)
            handlers.append(file_handler)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        for handler in handlers:
            handler.setFormatter(formatter)
            handler.setLevel(log_level)
            logger.addHandler(handler)

        if rank == 0:
            logger.setLevel(log_level)
        else:
            logger.setLevel(logging.ERROR)

        logger_initialized[name] = True

        return logger

    @staticmethod
    def get_root_logger(log_file=None, log_level=logging.INFO):
        logger = Utils.get_logger(
            name="DeepModel", log_file=log_file, log_level=log_level
        )

        return logger

    @staticmethod
    def console_logger_start():
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())

        os.makedirs("./work_dirs/", exist_ok=True)
        console_log_file = os.path.join("./work_dirs/", f"{timestamp}.log")
        console_logger = Utils.get_root_logger(
            log_file=console_log_file, log_level=logging.INFO
        )
        return console_logger


class PytorchLightningDefault(pl.LightningModule):
    def __init__(self, logger=None):
        super(PytorchLightningDefault, self).__init__()

        self.n_logger = logger
        self.console_logger = Utils.console_logger_start()
        # self.UsedUniDataloader = UniDataloader()
        self.val_scales = 1
        self.test_scales = 1
        self.ignore = False
        self.log_config_step = dict(
            on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True
        )

        self.log_config_epoch = dict(
            on_epoch=True, prog_bar=True, logger=True, sync_dist=True
        )

        self.module_lr_dict = dict(backbone=NetConfig.backbone_lr)

        (
            self.train_accuracy,
            self.train_precision,
            self.train_recall,
            self.val_accuracy,
            self.val_precision,
            self.val_recall,
            self.test_accuracy,
            self.test_precision,
            self.test_recall,
        ) = PytorchLightningDefault.get_classification_metrics(
            DataConfig.num_classes, self.ignore
        )

    @staticmethod
    def get_classification_metrics(num_classes, ignore):
        import torchmetrics

        if ignore:
            pass
        else:
            train_accuracy = torchmetrics.Accuracy(
                task="multiclass", num_classes=num_classes, average="none"
            )

            train_precision = torchmetrics.Precision(
                task="multiclass", num_classes=num_classes, average="none"
            )

            train_recall = torchmetrics.Recall(
                task="multiclass", num_classes=num_classes, average="none"
            )

            val_accuracy = torchmetrics.Accuracy(
                task="multiclass", num_classes=num_classes, average="none"
            )

            val_precision = torchmetrics.Precision(
                task="multiclass", num_classes=num_classes, average="none"
            )

            val_recall = torchmetrics.Recall(
                task="multiclass", num_classes=num_classes, average="none"
            )

            test_accuracy = torchmetrics.Accuracy(
                task="multiclass", num_classes=num_classes, average="none"
            )

            test_precision = torchmetrics.Precision(
                task="multiclass", num_classes=num_classes, average="none"
            )

            test_recall = torchmetrics.Recall(
                task="multiclass", num_classes=num_classes, average="none"
            )

        return (
            train_accuracy,
            train_precision,
            train_recall,
            val_accuracy,
            val_precision,
            val_recall,
            test_accuracy,
            test_precision,
            test_recall,
        )

    def train_dataloader(self):
        train_loader = GetImagesCIFAR10.get_cifar10_dataloader(
            batch_size=TrainingConfig.batch_size,
            data_dir=TrainingConfig.data_dir,
            data_category="train",
        )

        return train_loader

    def val_dataloader(self):
        val_loader = GetImagesCIFAR10.get_cifar10_dataloader(
            batch_size=ValidationConfig.batch_size,
            data_dir=TrainingConfig.data_dir,
            data_category="test",
        )

        return val_loader

    def lr_logging(self):
        """+++ Capture the learning rates and log it out using logger;"""
        lightning_optimizer = self.optimizers()
        param_groups = lightning_optimizer.optimizer.param_groups

        for param_group_idx in range(len(param_groups)):
            sub_param_group = param_groups[param_group_idx]

            """>>>
            print("==>> sub_param_group: ", sub_param_group.keys())
            # ==>> sub_param_group:  dict_keys(['params', 'lr', 'momentum', 'dampening', 'weight_decay', 'nesterov', 'maximize', 'initial_lr'])
            """

            sub_lr_name = "lr/lr_" + str(param_group_idx)
            """>>>
            print("lr: {}, {}".format(sub_lr_name, sub_param_group["lr"]))
            # lr: lr_0, 0.001
            # lr: lr_1, 0.08
            """

            self.log(
                sub_lr_name,
                sub_param_group["lr"],
                batch_size=TrainingConfig.batch_size,
                **self.log_config_step,
            )

    def configure_optimizers(self):
        optimizer = self.get_optim()

        if TrainingConfig.pretrained_weights:
            max_epochs = TrainingConfig.pretrained_weights_max_epoch
        else:
            max_epochs = TrainingConfig.max_epochs

        if TrainingConfig.scheduler == "cosineAnn":
            eta_min = TrainingConfig.eta_min

            T_max = max_epochs
            last_epoch = -1

            sch = CosineAnnealingLR(
                optimizer, T_max=T_max, eta_min=eta_min, last_epoch=last_epoch
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": sch, "monitor": "train_loss"},
            }

        elif TrainingConfig.scheduler == "CosineAnnealingLR":
            steps = 10
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": sch, "monitor": "train_loss"},
            }
        elif TrainingConfig.scheduler == "step":
            step_size = int(TrainingConfig.step_ratio * max_epochs)
            gamma = TrainingConfig.gamma
            sch = StepLR(optimizer, step_size=step_size, gamma=gamma)

            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": sch, "monitor": "train_loss"},
            }
        elif TrainingConfig.scheduler == "none":
            return optimizer

    def different_lr(self, module_lr_dict, lr):
        def is_key_included(module_name, n):
            return module_name in n

        def is_any_key_match(module_lr_dict, n):
            indicator = False
            for key in module_lr_dict.keys():
                if key in n:
                    indicator = True
            return indicator

        params = list(self.named_parameters())

        grouped_parameters = [
            {
                "params": [
                    p for n, p in params if not is_any_key_match(module_lr_dict, n)
                ],
                "lr": lr,
            }
        ]

        for key in module_lr_dict.keys():
            sub_param_list = []
            for n, p in params:
                if is_key_included(key, n):
                    if module_lr_dict[key] == 0.0:
                        p.requires_grad = False
                    sub_param_list.append(p)
            sub_parameters = {"params": sub_param_list, "lr": module_lr_dict[key]}
            grouped_parameters.append(sub_parameters)

        return grouped_parameters

    def get_optim(self):
        lr = NetConfig.lr

        if TrainingConfig.pretrained_weights:
            lr = TrainingConfig.pre_lr

        if not hasattr(torch.optim, NetConfig.opt):
            print("Optimiser {} not supported".format(NetConfig.opt))
            raise NotImplementedError

        optim = getattr(torch.optim, NetConfig.opt)

        if TrainingConfig.strategy == "colossalai":
            from colossalai.nn.optimizer import HybridAdam

            optimizer = HybridAdam(self.parameters(), lr=lr)
        else:
            if TrainingConfig.single_lr:
                print("Using a single learning rate for all parameters")
                grouped_parameters = [{"params": self.parameters()}]
            else:
                print("Using different learning rates for all parameters")
                grouped_parameters = self.different_lr(self.module_lr_dict, lr)

            # print("\n ==>> grouped_parameters: \n", grouped_parameters)

            if NetConfig.opt == "Adam":
                lr = 0.001
                betas = (0.9, 0.999)
                eps = 1e-08
                weight_decay = 0.0

                optimizer = torch.optim.Adam(
                    grouped_parameters,
                    lr=lr,
                    betas=betas,
                    eps=1e-08,
                    weight_decay=weight_decay,
                )
            elif NetConfig.opt == "Lamb":
                weight_decay = 0.02
                betas = (0.9, 0.999)

                optimizer = torch.optim.Lamb(
                    grouped_parameters, lr=lr, betas=betas, weight_decay=weight_decay
                )
            elif NetConfig.opt == "AdamW":
                eps = 1e-8
                betas = (0.9, 0.999)
                weight_decay = 0.05

                optimizer = torch.optim.AdamW(
                    grouped_parameters,
                    lr=lr,
                    betas=betas,
                    eps=eps,
                    weight_decay=weight_decay,
                )
            elif NetConfig.opt == "SGD":
                momentum = 0.9
                weight_decay = 0.0001

                optimizer = torch.optim.SGD(
                    grouped_parameters,
                    lr=lr,
                    momentum=momentum,
                    weight_decay=weight_decay,
                )

            else:
                optimizer = optim(grouped_parameters, lr=NetConfig.lr)

        optimizer.zero_grad()

        return optimizer


class Model(PytorchLightningDefault):
    def __init__(self, logger=None):
        super(Model, self).__init__()

        self.module_lr_dict = dict(placeholder=0.0)

        init_model, backbone, init_classifier, classifier = TorchModel.get_model(
            num_classes=DataConfig.num_classes
        )
        # TorchModel.check_weights_loading(init_model)

        # init_model_bk = nn.Sequential()
        # init_model_bk.add_module("backbone", backbone)
        # init_model_bk.add_module("init_classifier", init_classifier)
        # # init_model_bk = init_model_bk.to(TrainingConfig.device)

        # TorchModel.check_weights_loading(init_model_bk)

        self.model = nn.Sequential()
        self.model.add_module("backbone", backbone)
        self.model.add_module("classifier", classifier)
        # model = model.to(TrainingConfig.device)

        self.loss = nn.CrossEntropyLoss()

    def forward(self, images, labels=None, epoch=None, batch_idx=None):
        output = self.model(images)

        return output

    def training_step(self, batch, batch_idx):
        self.lr_logging()

        images, gt_labels = batch

        model_outputs = self.forward(images)
        train_loss = self.loss(model_outputs, gt_labels)

        # train_loss = train_losses_dict["loss"]

        losses = {"loss": train_loss}

        self.log(
            "train_loss",
            train_loss,
            batch_size=TrainingConfig.batch_size,
            **self.log_config_step,
        )

        model_predictions = model_outputs.argmax(dim=1)

        self.train_accuracy.update(model_predictions, gt_labels)
        self.train_precision.update(model_predictions, gt_labels)
        self.train_recall.update(model_predictions, gt_labels)

        if batch_idx % 10 == 0:
            self.console_logger.info(
                "epoch: {0:04d} | loss_train: {1:.4f}".format(
                    self.current_epoch, losses["loss"]
                )
            )

        return {"loss": losses["loss"]}

    def training_epoch_end(self, outputs):
        cwd = os.getcwd()
        print("==>> Expriment Folder: ", cwd)

        train_accuracy = self.train_accuracy.compute()

        """>>> If self.ignore is used, the last clss is fake and will not calculated in the metric; """
        if self.ignore:
            train_accuracy_mean = torch.mean(train_accuracy[:-1]).item()
        else:
            train_accuracy_mean = torch.mean(train_accuracy).item()

        self.log(
            "train_accuracy_epoch",
            train_accuracy_mean,
            batch_size=TrainingConfig.batch_size,
            **self.log_config_epoch,
        )

        self.train_accuracy.reset()
        self.train_precision.reset()
        self.train_recall.reset()

    def validation_step(self, batch, batch_idx):
        images, gt_labels = batch

        model_outputs = self.forward(images)
        val_loss = self.loss(model_outputs, gt_labels)
        # val_loss = val_losses_dict["loss"]

        model_predictions = model_outputs.argmax(dim=1)

        self.val_accuracy.update(model_predictions, gt_labels)
        self.val_precision.update(model_predictions, gt_labels)
        self.val_recall.update(model_predictions, gt_labels)

        self.log(
            "val_loss",
            val_loss,
            batch_size=ValidationConfig.batch_size * self.val_scales,
            **self.log_config_step,
        )

        return val_loss

    def validation_epoch_end(self, outputs):
        """>>> The compute() function will return a list;"""
        val_epoch_accuracy = self.val_accuracy.compute()
        val_epoch_precision = self.val_precision.compute()
        val_epoch_recall = self.val_recall.compute()

        if self.ignore:
            val_epoch_accuracy_mean = torch.mean(val_epoch_accuracy[:-1].item())
            val_epoch_precision_mean = torch.mean(val_epoch_precision[:-1]).item()
            val_epoch_recall_mean = torch.mean(val_epoch_recall[:-1]).item()
        else:
            val_epoch_accuracy_mean = torch.mean(val_epoch_accuracy).item()
            val_epoch_precision_mean = torch.mean(val_epoch_precision).item()
            val_epoch_recall_mean = torch.mean(val_epoch_recall).item()

        self.log(
            "val_epoch_accuracy",
            val_epoch_accuracy_mean,
            batch_size=ValidationConfig.batch_size * self.val_scales,
            **self.log_config_epoch,
        )

        """>>> We use the "user_metric" variable to monitor the performance on val set; """
        user_metric = val_epoch_accuracy_mean
        self.log(
            "user_metric",
            user_metric,
            batch_size=ValidationConfig.batch_size * self.val_scales,
            **self.log_config_epoch,
        )

        """>>> Plot the results to console using only one gpu since the results on all gpus are the same; """
        if self.global_rank == 0:
            self.console_logger.info("epoch: {0:04d} ".format(self.current_epoch))

            for i in range(DataConfig.num_classes):
                self.console_logger.info(
                    "{0: <15}, acc: {1:.4f} | precision: {2:.4f} | recall: {3:.4f}".format(
                        DataConfig.classes[i],
                        val_epoch_accuracy[i].item(),
                        val_epoch_precision[i].item(),
                        val_epoch_recall[i].item(),
                    )
                )
            self.console_logger.info(
                "acc_mean: {0:.4f} ".format(val_epoch_accuracy_mean)
            )

            self.console_logger.info(
                "precision_mean: {0:.4f} ".format(val_epoch_precision_mean)
            )
            self.console_logger.info(
                "recall_mean: {0:.4f} ".format(val_epoch_recall_mean)
            )

        self.val_accuracy.reset()
        self.val_precision.reset()
        self.val_recall.reset()


class ExprCommonSetting:
    @staticmethod
    def generate_folders():
        import os

        data_folder = "./data/"
        work_dirs_folder = "./work_dirs/"

        os.makedirs(data_folder, exist_ok=True)
        os.makedirs(work_dirs_folder, exist_ok=True)


class ExprSetting(object):
    @staticmethod
    def logger_setting():
        lr_logger = LearningRateMonitor(logging_interval="epoch")

        if TrainingConfig.logger_name == "neptune":
            print("Not implemented")
            sys.exit()
        elif TrainingConfig.logger_name == "csv":
            print_logger = CSVLogger(DataConfig.logger_root)
        elif TrainingConfig.logger_name == "wandb":
            run_name = datetime.datetime.now(tz=pytz.timezone("US/Central")).strftime(
                "%Y-%m-%d-%H-%M"
            )
            print_logger = WandbLogger(
                project=TrainingConfig.wandb_name,
                settings=wandb.Settings(code_dir="."),
                name=run_name,
            )
        else:
            print_logger = CSVLogger(DataConfig.logger_root)
        return lr_logger, print_logger

    @staticmethod
    def checkpoint_setting():
        model_checkpoint = ModelCheckpoint(
            filename="{epoch}-{val_loss:.2f}-{user_metric:.2f}",
            save_last=True,
            save_weights_only=True,
            save_top_k=1,
            monitor="val_loss",
            mode="min",
        )
        return model_checkpoint

    @staticmethod
    def gpu_setting():
        if TrainingConfig.training_mode == False:
            num_gpus = 1
        else:
            if isinstance(TrainingConfig.num_gpus, int):
                num_gpus = TrainingConfig.num_gpus
            elif TrainingConfig.num_gpus == "autocount":
                TrainingConfig.num_gpus = torch.cuda.device_count()
                num_gpus = TrainingConfig.num_gpus
            else:
                gpu_list = TrainingConfig.num_gpus.split(",")
                num_gpus = len(gpu_list)
        print("\n num of gpus: {} \n".format(num_gpus))

        return num_gpus

    @staticmethod
    def dataset_setting():
        seed_everything(DataConfig.random_seed)
        if DataConfig.classes is None:
            DataConfig.classes = np.arange(DataConfig.num_classes)
        return DataConfig

    @staticmethod
    def earlystop_setting():
        """+++ currently, disable the earlstopping settings;"""
        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=2000000,
            strict=False,
            verbose=False,
            mode="min",
        )
        return early_stop

    # @staticmethod
    # def dynamic_dataloaders():
    #     """ +++ all the datasets should have the same dataloader name; """
    #     if DataConfig.dataloader_name == "dataloader":
    #         from dataloader import UniDataloader
    #     else:
    #         sys.eixt("Please check your dataloader name in config file ... ")

    #     UsedUniDataloader = UniDataloader()
    #     return UsedUniDataloader

    # @staticmethod
    # def dynamic_models():
    #     """ +++ make sure all the models have the same model name --- model; """
    #     if NetConfig.model_name == "model":
    #         from model import Model
    #     else:
    #         sys.eixt("Please check your model name in config file ... ")

    #     UniModel = Model
    #     return UniModel

    @staticmethod
    def pl_trainer_setting():
        if TrainingConfig.cpus:
            trainer = pl.Trainer(
                num_nodes=TrainingConfig.num_nodes,
                precision=TrainingConfig.precision,
                accelerator="cpu",
                strategy=DDPStrategy(find_unused_parameters=True),
                # profiler="pytorch",
                logger=ExprSetting.logger_setting()[1],
                callbacks=[
                    ExprSetting.logger_setting()[0],
                    ExprSetting.checkpoint_setting(),
                ],
                log_every_n_steps=1,
                check_val_every_n_epoch=ValidationConfig.val_interval,
                auto_scale_batch_size="binsearch",
                replace_sampler_ddp=False,
            )
            return trainer
        else:
            if (
                TrainingConfig.strategy == "ddp"
                or TrainingConfig.strategy == "ddp_sharded"
            ):
                trainer = pl.Trainer(
                    devices=ExprSetting.gpu_setting(),
                    num_nodes=TrainingConfig.num_nodes,
                    precision=TrainingConfig.precision,
                    accelerator=TrainingConfig.accelerator,
                    strategy=TrainingConfig.strategy,
                    logger=ExprSetting.logger_setting()[1],
                    callbacks=[
                        ExprSetting.logger_setting()[0],
                        ExprSetting.checkpoint_setting(),
                    ],
                    log_every_n_steps=1,
                    max_epochs=TrainingConfig.max_epochs,
                    check_val_every_n_epoch=ValidationConfig.val_interval,
                    auto_scale_batch_size="binsearch",
                    # resume_from_checkpoint="",
                    # sync_batchnorm=True if num_gpus > 1 else False,
                    # plugins=DDPPlugin(find_unused_parameters=False),
                    # track_grad_norm=1,
                    # progress_bar_refresh_rate=TrainingConfig.progress_bar_refresh_rate,
                    # profiler="pytorch",  # "simple", "advanced","pytorch"
                )
                return trainer
            elif TrainingConfig.strategy == "deepspeed":
                deepspeed_strategy = DeepSpeedStrategy(
                    offload_optimizer=True,
                    allgather_bucket_size=5e8,
                    reduce_bucket_size=5e8,
                )

                trainer = pl.Trainer(
                    devices=ExprSetting.gpu_setting(),
                    num_nodes=TrainingConfig.num_nodes,
                    precision=TrainingConfig.precision,
                    accelerator=TrainingConfig.accelerator,
                    strategy=deepspeed_strategy,
                    logger=ExprSetting.logger_setting()[1],
                    callbacks=[
                        ExprSetting.logger_setting()[0],
                        ExprSetting.checkpoint_setting(),
                    ],
                    log_every_n_steps=1,
                    max_epochs=TrainingConfig.max_epochs,
                    check_val_every_n_epoch=ValidationConfig.val_interval,
                    auto_scale_batch_size="binsearch",
                )
                return trainer
            elif TrainingConfig.strategy == "bagua":
                bagua_strategy = BaguaStrategy(
                    algorithm=TrainingConfig.bagua_sub_strategy
                )  # "gradient_allreduce"; bytegrad"; "decentralized"; "low_precision_decentralized"; qadam"; async";

                trainer = pl.Trainer(
                    devices=ExprSetting.gpu_setting(),
                    num_nodes=TrainingConfig.num_nodes,
                    precision=TrainingConfig.precision,
                    accelerator=TrainingConfig.accelerator,
                    strategy=bagua_strategy,
                    logger=ExprSetting.logger_setting()[1],
                    callbacks=[
                        ExprSetting.logger_setting()[0],
                        ExprSetting.checkpoint_setting(),
                    ],
                    log_every_n_steps=1,
                    max_epochs=TrainingConfig.max_epochs,
                    check_val_every_n_epoch=ValidationConfig.val_interval,
                    auto_scale_batch_size="binsearch",
                )
                return trainer
            else:
                print("\n Trainer configuration is wrong. \n")


class GetImagesCIFAR10:
    """+++ use cifar10 as an example;"""

    @staticmethod
    def get_cifar10_dataloader(batch_size, data_dir, data_category="train"):
        import torch
        import torchvision
        import torchvision.transforms as transforms

        if data_category == "train":
            use_training = True
            use_shuffle = True
        else:
            use_training = False
            use_shuffle = False

        transform = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=0),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        dataset = torchvision.datasets.CIFAR10(
            root=data_dir,
            train=use_training,
            download=True,
            transform=transform,
        )

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=use_shuffle,
            num_workers=DataConfig.workers,
            pin_memory=DataConfig.pin_memory,
        )

        return loader


class DataConfig:
    logger_root = "./work_dirs/main_logger/"
    work_dirs = "./work_dirs/"
    random_seed = 42
    workers = 16
    pin_memory = True

    num_classes = 10
    dataset_name = "cifar10"
    classes = None

    if classes is None:
        classes = np.arange(num_classes)


class NetConfig:
    backbone_name = "efficientnet"
    lr = 0.08
    opt = "SGD"
    backbone_lr = 0.01


class TrainingConfig:
    batch_size = 32 * 8
    data_dir = "./data/cifar10_online/"
    # data_category = "train"
    num_epochs = 10
    print_interval = 20
    # num_classes = 10
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

    wandb_name = "pt-" + DataConfig.dataset_name + "-" + NetConfig.backbone_name

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
    max_epochs = T_0 + T_mult * T_0


class ValidationConfig:
    val_interval = 1
    batch_size = 16


if __name__ == "__main__":
    ExprCommonSetting.generate_folders()

    """ +++ check model loading process. make sure it is correct."""
    # init_model, backbone, init_classifier, classifier = TorchModel.get_model(
    #     num_classes=DataConfig.num_classes
    # )
    # TorchModel.check_weights_loading(init_model)

    # init_model_bk = nn.Sequential()
    # init_model_bk.add_module("backbone", backbone)
    # init_model_bk.add_module("init_classifier", init_classifier)
    # # init_model_bk = init_model_bk.to(TrainingConfig.device)

    # TorchModel.check_weights_loading(init_model_bk)

    model = Model()

    pl_trainer = ExprSetting.pl_trainer_setting()

    if TrainingConfig.training_mode:
        pl_trainer.fit(model)
    else:
        pl_trainer.validate(model, dataloaders=model.val_dataloader())
        # pl_trainer.test(model, dataloaders=val_dataloader)
