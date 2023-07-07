import datetime
import sys

import numpy as np
import pytorch_lightning as pl
import pytz
import torch
import wandb
from c5_configs import Configs
from c2_data import DataModule
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.strategies import BaguaStrategy, DDPStrategy
from pytorch_lightning.strategies.deepspeed import DeepSpeedStrategy


class SettingUtils:
    @staticmethod
    def generate_folders():
        import os

        data_folder = "./data/"
        work_dirs_folder = "./work_dirs/"

        os.makedirs(data_folder, exist_ok=True)
        os.makedirs(work_dirs_folder, exist_ok=True)


class Settings(object):
    @staticmethod
    def logger_setting():
        lr_logger = LearningRateMonitor(logging_interval="epoch")

        if Configs.logger_name == "neptune":
            print("Not implemented")
            sys.exit()
        elif Configs.logger_name == "csv":
            print_logger = CSVLogger(Configs.logger_root)
        elif Configs.logger_name == "wandb":
            run_name = datetime.datetime.now(tz=pytz.timezone("US/Central")).strftime(
                "%Y-%m-%d-%H-%M"
            )
            print_logger = WandbLogger(
                project=Configs.wandb_name,
                settings=wandb.Settings(code_dir="."),
                name=run_name,
            )
        else:
            print_logger = CSVLogger(Configs.logger_root)
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
        if Configs.training_mode == False:
            num_gpus = 1
        else:
            if isinstance(Configs.num_gpus, int):
                num_gpus = Configs.num_gpus
            elif Configs.num_gpus == "autocount":
                Configs.num_gpus = torch.cuda.device_count()
                num_gpus = Configs.num_gpus
            else:
                gpu_list = Configs.num_gpus.split(",")
                num_gpus = len(gpu_list)
        print("\n num of gpus: {} \n".format(num_gpus))

        return num_gpus

    @staticmethod
    def dataset_setting():
        seed_everything(Configs.random_seed)
        if Configs.classes is None:
            Configs.classes = np.arange(Configs.num_classes)
        return Configs

    @staticmethod
    def data_module_setting():
        data_module = DataModule()
        return data_module

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

    @staticmethod
    def pl_trainer_setting(num_gpus):
        if Configs.cpus:
            trainer = pl.Trainer(
                num_nodes=Configs.num_nodes,
                precision=Configs.precision,
                accelerator="cpu",
                strategy=DDPStrategy(find_unused_parameters=True),
                # profiler="pytorch",
                logger=Settings.logger_setting()[1],
                callbacks=[
                    Settings.logger_setting()[0],
                    Settings.checkpoint_setting(),
                ],
                log_every_n_steps=1,
                check_val_every_n_epoch=Configs.val_interval,
                auto_scale_batch_size="binsearch",
                replace_sampler_ddp=False,
            )
            return trainer
        else:
            if Configs.strategy == "ddp" or Configs.strategy == "ddp_sharded":
                trainer = pl.Trainer(
                    devices=num_gpus,
                    num_nodes=Configs.num_nodes,
                    precision=Configs.precision,
                    accelerator=Configs.accelerator,
                    # strategy=Configs.strategy,
                    logger=Settings.logger_setting()[1],
                    callbacks=[
                        Settings.logger_setting()[0],
                        Settings.checkpoint_setting(),
                    ],
                    log_every_n_steps=1,
                    max_epochs=Configs.max_epochs,
                    check_val_every_n_epoch=Configs.val_interval,
                    auto_scale_batch_size=True,
                    # find_unused_parameters=False,
                    # resume_from_checkpoint="",
                    # sync_batchnorm=True if num_gpus > 1 else False,
                    plugins=DDPPlugin(find_unused_parameters=False),
                    # track_grad_norm=1,
                    # progress_bar_refresh_rate=Configs.progress_bar_refresh_rate,
                    # profiler="pytorch",  # "simple", "advanced","pytorch"
                )
                return trainer
            elif Configs.strategy == "deepspeed":
                deepspeed_strategy = DeepSpeedStrategy(
                    offload_optimizer=True,
                    allgather_bucket_size=5e8,
                    reduce_bucket_size=5e8,
                )

                trainer = pl.Trainer(
                    devices=num_gpus,
                    num_nodes=Configs.num_nodes,
                    precision=Configs.precision,
                    accelerator=Configs.accelerator,
                    strategy=deepspeed_strategy,
                    logger=Settings.logger_setting()[1],
                    callbacks=[
                        Settings.logger_setting()[0],
                        Settings.checkpoint_setting(),
                    ],
                    log_every_n_steps=1,
                    max_epochs=Configs.max_epochs,
                    check_val_every_n_epoch=Configs.val_interval,
                    auto_scale_batch_size="binsearch",
                )
                return trainer
            elif Configs.strategy == "bagua":
                bagua_strategy = BaguaStrategy(
                    algorithm=Configs.bagua_sub_strategy
                )  # "gradient_allreduce"; bytegrad"; "decentralized"; "low_precision_decentralized"; qadam"; async";

                trainer = pl.Trainer(
                    devices=num_gpus,
                    num_nodes=Configs.num_nodes,
                    precision=Configs.precision,
                    accelerator=Configs.accelerator,
                    strategy=bagua_strategy,
                    logger=Settings.logger_setting()[1],
                    callbacks=[
                        Settings.logger_setting()[0],
                        Settings.checkpoint_setting(),
                    ],
                    log_every_n_steps=1,
                    max_epochs=Configs.max_epochs,
                    check_val_every_n_epoch=Configs.val_interval,
                    auto_scale_batch_size="binsearch",
                )
                return trainer
            else:
                print("\n Trainer configuration is wrong. \n")
