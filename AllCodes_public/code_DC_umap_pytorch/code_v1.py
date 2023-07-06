import datetime
import logging
import multiprocessing
import os
import sys
import time
import matplotlib.pyplot as plt

import numpy as np
import pytorch_lightning as pl
import pytz
import timm
import torch
import torch.distributed as dist
import torch.nn as nn
import wandb
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from pynndescent import NNDescent
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
from sklearn.neighbors import KDTree
from sklearn.utils import check_random_state
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from umap.spectral import spectral_layout
from umap.umap_ import find_ab_params, fuzzy_simplicial_set


class ModelUtils(pl.LightningModule):
    def __init__(self, logger=None):
        super(ModelUtils, self).__init__()

        self.n_logger = logger
        self.console_logger = Utils.console_logger_start()
        self.val_scales = 1
        self.test_scales = 1
        self.ignore = False
        self.batch_size = Configs.batch_size
        self.log_config_step = dict(
            on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True
        )

        self.log_config_epoch = dict(
            on_epoch=True, prog_bar=True, logger=True, sync_dist=True
        )

        self.module_lr_dict = dict(backbone=Configs.backbone_lr)

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
        ) = ModelUtils.get_classification_metrics(Configs.num_classes, self.ignore)

        self.save_hyperparameters()

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

    def lr_logging(self):
        """+++ Capture the learning rates and log it out using logger;"""
        lightning_optimizer = self.optimizers()
        param_groups = lightning_optimizer.optimizer.param_groups

        for param_group_idx in range(len(param_groups)):
            sub_param_group = param_groups[param_group_idx]
            sub_lr_name = "lr/lr_" + str(param_group_idx)

            self.log(
                sub_lr_name,
                sub_param_group["lr"],
                batch_size=Configs.batch_size,
                **self.log_config_step,
            )

    def configure_optimizers(self):
        optimizer = self.get_optim()

        if Configs.pretrained_weights:
            max_epochs = Configs.pretrained_weights_max_epoch
        else:
            max_epochs = Configs.max_epochs

        if Configs.scheduler == "cosineAnn":
            eta_min = Configs.eta_min

            T_max = max_epochs
            last_epoch = -1

            sch = CosineAnnealingLR(
                optimizer, T_max=T_max, eta_min=eta_min, last_epoch=last_epoch
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": sch, "monitor": "train_loss"},
            }

        elif Configs.scheduler == "CosineAnnealingLR":
            steps = 10
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": sch, "monitor": "train_loss"},
            }
        elif Configs.scheduler == "step":
            step_size = int(Configs.step_ratio * max_epochs)
            gamma = Configs.gamma
            sch = StepLR(optimizer, step_size=step_size, gamma=gamma)

            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": sch, "monitor": "train_loss"},
            }
        elif Configs.scheduler == "none":
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
        lr = Configs.lr

        if Configs.pretrained_weights:
            lr = Configs.pre_lr

        if not hasattr(torch.optim, Configs.opt):
            print("Optimiser {} not supported".format(Configs.opt))
            raise NotImplementedError

        optim = getattr(torch.optim, Configs.opt)

        if Configs.strategy == "colossalai":
            from colossalai.nn.optimizer import HybridAdam

            optimizer = HybridAdam(self.parameters(), lr=lr)
        else:
            if Configs.single_lr:
                print("Using a single learning rate for all parameters")
                grouped_parameters = [{"params": self.parameters()}]
            else:
                print("Using different learning rates for all parameters")
                grouped_parameters = self.different_lr(self.module_lr_dict, lr)

            # print("\n ==>> grouped_parameters: \n", grouped_parameters)

            if Configs.opt == "Adam":
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
            elif Configs.opt == "Lamb":
                weight_decay = 0.02
                betas = (0.9, 0.999)

                optimizer = torch.optim.Lamb(
                    grouped_parameters, lr=lr, betas=betas, weight_decay=weight_decay
                )
            elif Configs.opt == "AdamW":
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
            elif Configs.opt == "SGD":
                momentum = 0.9
                weight_decay = 0.0001

                optimizer = torch.optim.SGD(
                    grouped_parameters,
                    lr=lr,
                    momentum=momentum,
                    weight_decay=weight_decay,
                )

            else:
                optimizer = optim(grouped_parameters, lr=Configs.lr)

        optimizer.zero_grad()

        return optimizer

    @staticmethod
    def get_model(num_classes):
        import timm

        keyworlds = "resnet50"
        selected_listed_models = ModelUtils.get_timm_model_list(keyworlds)
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
    def get_vit_model():
        vit_model = ViT(
            image_size=224,
            patch_size=8,
            num_classes=10,
            dim=128,
            depth=8,
            heads=8,
            mlp_dim=384,
            pool="cls",
            channels=3,
            dim_head=64,
            dropout=0.0,
            emb_dropout=0.0,
        )

        return vit_model

    @staticmethod
    def get_vit_timm(num_classes):
        import timm

        keyworlds = "vit"
        selected_listed_models = ModelUtils.get_timm_model_list(keyworlds)
        # print(f"==>> selected_listed_models: {selected_listed_models}")

        model_name = "vit_base_patch8_224"  # "vit_small_patch32_224"
        init_model = timm.create_model(model_name, pretrained=True)
        feature_dim = init_model.get_classifier().in_features

        backbone = nn.Sequential()
        init_classifier = nn.Sequential()
        named_modules_list = list(init_model.named_children())

        # print("\n timm module list: ")
        # module_name_list = [name for name, _ in named_modules_list]
        # print(f"==>> module_name_list: {module_name_list} \n")

        # for name, moudule in named_modules_list:
        #     print(f"==>> name: {name}, moudule: {moudule}")
        # print("\n ")

        for i, (name, module) in enumerate(init_model.named_children()):
            if i >= len(named_modules_list) - 1:
                init_classifier.add_module(name, module)
                # break
            else:
                backbone.add_module(name, module)

        if num_classes == 1000:
            classifier = init_classifier
        else:
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
    def get_clustering_encoder(data, graph):
        model = CodeSpaceTable(
            continuous=Configs.continuous,
            n_points=Configs.num_samples,
            dim=Configs.dim,
        ).cuda()

        init_embedding = ModelUtils.init_embedding_from_graph(
            data.numpy(),
            graph,
            Configs.dim,
            random_state=None,
            metric="euclidean",
            _metric_kwds={},
            init="spectral",
        )

        model.code_space.data = torch.from_numpy(init_embedding).cuda()

        return model

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
            input_batch = input_batch.to(Configs.device)
            to_test_model.to(Configs.device)

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

    @staticmethod
    def convert_distance_to_log_probability(distances, a=1.0, b=1.0):
        """
        convert distance representation into log probability,
            as a function of a, b params

        Parameters
        ----------
        distances : array
            euclidean distance between two points in embedding
        a : float, optional
            parameter based on min_dist, by default 1.0
        b : float, optional
            parameter based on min_dist, by default 1.0

        Returns
        -------
        float
            log probability in embedding space
        """
        return -torch.log1p(a * distances ** (2 * b))

    @staticmethod
    def init_embedding_from_graph(
        _raw_data,
        graph,
        n_components,
        random_state,
        metric,
        _metric_kwds,
        init="spectral",
    ):
        """Initialize embedding using graph. This is for direct embeddings.

        Parameters
        ----------
        init : str, optional
            Type of initialization to use. Either random, or spectral, by default "spectral"

        Returns
        -------
        embedding : np.array
            the initialized embedding
        """

        if random_state is None:
            random_state = check_random_state(None)

        if isinstance(init, str) and init == "random":
            embedding = random_state.uniform(
                low=-10.0, high=10.0, size=(graph.shape[0], n_components)
            ).astype(np.float32)
        elif isinstance(init, str) and init == "spectral":
            # We add a little noise to avoid local minima for optimization to come

            initialisation = spectral_layout(
                _raw_data,
                graph,
                n_components,
                random_state,
                metric=metric,
                metric_kwds=_metric_kwds,
            )
            expansion = 10.0 / np.abs(initialisation).max()
            embedding = (initialisation * expansion).astype(
                np.float32
            ) + random_state.normal(
                scale=0.0001, size=[graph.shape[0], n_components]
            ).astype(
                np.float32
            )

        else:
            init_data = np.array(init)
            if len(init_data.shape) == 2:
                if np.unique(init_data, axis=0).shape[0] < init_data.shape[0]:
                    tree = KDTree(init_data)
                    dist, ind = tree.query(init_data, k=2)
                    nndist = np.mean(dist[:, 1])
                    embedding = init_data + random_state.normal(
                        scale=0.001 * nndist, size=init_data.shape
                    ).astype(np.float32)
                else:
                    embedding = init_data

        return embedding

    @staticmethod
    def random_sampled_embedding_pairs(edges_to_exp, edges_from_exp):
        import random

        indices = list(np.arange(edges_to_exp.shape[0]))
        sampled_indices = random.sample(indices, Configs.batch_size)
        sampled_edges_to = edges_to_exp[sampled_indices]
        sampled_edges_from = edges_from_exp[sampled_indices]

        return sampled_edges_to, sampled_edges_from


# class CodeSpaceTable(torch.nn.Module):
#     def __init__(self, continuous, n_points, dim):
#         super(CodeSpaceTable, self).__init__()
#         self.continuous = continuous
#         self.code_space = torch.nn.Parameter(torch.randn(n_points, dim) * 0.1)

#     def forward(
#         self,
#     ):
#         if self.continuous:
#             return self.code_space
#         else:
#             return torch.nn.functional.log_softmax(self.code_space, 1)


class Encoder(nn.Module):
    def __init__(self, dims, n_components=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.product(dims), 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, n_components),
        )

    def forward(self, X):
        return self.encoder(X)


class Models(ModelUtils):
    def __init__(self, data_torch, labels_torch, logger=None):
        super(Models, self).__init__()

        self.module_lr_dict = dict(placeholder=0.0)

        # init_model, backbone, init_classifier, classifier = ModelUtils.get_model(
        #     num_classes=Configs.num_classes
        # )
        # # ModelUtils.check_weights_loading(init_model)

        # # init_model_bk = nn.Sequential()
        # # init_model_bk.add_module("backbone", backbone)
        # # init_model_bk.add_module("init_classifier", init_classifier)
        # # # init_model_bk = init_model_bk.to(Configs.device)

        # # ModelUtils.check_weights_loading(init_model_bk)

        # self.model = nn.Sequential()
        # self.model.add_module("backbone", backbone)
        # self.model.add_module("classifier", classifier)
        # # model = model.to(Configs.device)

        """ +++ get vit from scratch."""
        # self.model = ModelUtils.get_vit_model()

        """ +++ get vit from timm."""
        # init_model, backbone, init_classifier, classifier = ModelUtils.get_vit_timm(
        #     Configs.num_classes
        # )

        # self.mean_module = MeanModule()
        # self.model = nn.Sequential()
        # self.model.add_module("backbone", backbone)
        # self.model.add_module("mean_vit_out", self.mean_module)
        # self.model.add_module("classifier", classifier)
        # # model = model.to(Configs.device)

        # self.model = ModelUtils.get_clustering_encoder(data, graph)

        # self.loss = nn.CrossEntropyLoss()

        # self.data_numpy = data_torch
        self.data_torch = data_torch.to(self.device)
        print(f"==>> self.data_torch.shape: {self.data_torch.shape}")
        self.labels_numpy = labels_torch.cpu().numpy()

        # self.model = CodeSpaceTable(
        #     continuous=Configs.continuous,
        #     n_points=self.data_numpy.shape[0],
        #     dim=Configs.dim,
        # )

        self.model = Encoder(dims=self.data_torch.shape[1:]).to(self.device)

        # self.init_embedding = Models.init_embedding_from_graph(
        #     self.data_torch.cpu().numpy(),
        #     graph,
        #     Configs.dim,
        #     random_state=None,
        #     metric="euclidean",
        #     _metric_kwds={},
        #     init="spectral",
        # )

        self.logsigmoid = torch.nn.LogSigmoid()

        self.batch_size = Configs.batch_size
        self.save_hyperparameters()

    @staticmethod
    def init_embedding_from_graph(
        _raw_data,
        graph,
        n_components,
        random_state,
        metric,
        _metric_kwds,
        init="spectral",
    ):
        """Initialize embedding using graph. This is for direct embeddings.

        Parameters
        ----------
        init : str, optional
            Type of initialization to use. Either random, or spectral, by default "spectral"

        Returns
        -------
        embedding : np.array
            the initialized embedding
        """

        if random_state is None:
            random_state = check_random_state(None)

        if isinstance(init, str) and init == "random":
            embedding = random_state.uniform(
                low=-10.0, high=10.0, size=(graph.shape[0], n_components)
            ).astype(np.float32)
        elif isinstance(init, str) and init == "spectral":
            # We add a little noise to avoid local minima for optimization to come

            initialisation = spectral_layout(
                _raw_data,
                graph,
                n_components,
                random_state,
                metric=metric,
                metric_kwds=_metric_kwds,
            )
            expansion = 10.0 / np.abs(initialisation).max()
            embedding = (initialisation * expansion).astype(
                np.float32
            ) + random_state.normal(
                scale=0.0001, size=[graph.shape[0], n_components]
            ).astype(
                np.float32
            )

        else:
            init_data = np.array(init)
            if len(init_data.shape) == 2:
                if np.unique(init_data, axis=0).shape[0] < init_data.shape[0]:
                    tree = KDTree(init_data)
                    dist, ind = tree.query(init_data, k=2)
                    nndist = np.mean(dist[:, 1])
                    embedding = init_data + random_state.normal(
                        scale=0.001 * nndist, size=init_data.shape
                    ).astype(np.float32)
                else:
                    embedding = init_data

        return embedding

    @staticmethod
    def random_sampled_embedding_pairs(edges_to_exp, edges_from_exp):
        import random

        indices = list(np.arange(edges_to_exp.shape[0]))
        sampled_indices = random.sample(indices, Configs.batch_size)
        sampled_edges_to = edges_to_exp[sampled_indices]
        sampled_edges_from = edges_from_exp[sampled_indices]

        return sampled_edges_to, sampled_edges_from

    # def forward(self, edges_to_exp, edges_from_exp, labels=None, epoch=None, batch_idx=None):
    #     # output = self.model(images)

    #     return output

    def training_step(self, batch, batch_idx):
        self.lr_logging()

        # images, gt_labels = batch

        # model_outputs = self.forward(images)
        # train_loss = self.loss(model_outputs, gt_labels)

        # # train_loss = train_losses_dict["loss"]

        edge_to_exp_with_data, edge_from_exp_with_data = batch

        edge_to_exp_embeddings = self.model(edge_to_exp_with_data)
        edge_from_exp_embeddings = self.model(edge_from_exp_with_data)

        # embedding_to_batch = embeddings[edge_to_exp]
        # embedding_from_batch = embeddings[edge_from_exp]
        embedding_neg_to = torch.repeat_interleave(
            edge_to_exp_embeddings, Configs.negative_sample_rate, axis=0
        )
        repeat_neg = torch.repeat_interleave(
            edge_from_exp_embeddings, Configs.negative_sample_rate, axis=0
        )
        embedding_neg_from = repeat_neg[torch.randperm(repeat_neg.shape[0])]

        distance_embedding = torch.concat(
            [
                torch.norm(edge_to_exp_embeddings - edge_from_exp_embeddings, dim=1),
                torch.norm(embedding_neg_to - embedding_neg_from, dim=1),
            ]
        )
        log_probabilities_distance = ModelUtils.convert_distance_to_log_probability(
            distance_embedding, Configs._a, Configs._b
        )

        probabilities_graph = torch.concat(
            [
                torch.ones(edge_to_exp_embeddings.shape[0]),
                torch.zeros(embedding_neg_to.shape[0]),
            ]
        ).to(self.device)

        attraction_term = -probabilities_graph * self.logsigmoid(
            log_probabilities_distance
        )
        # use numerically stable repellent term
        # Shi et al. 2022 (https://arxiv.org/abs/2111.08851)
        # log(1 - sigmoid(logits)) = log(sigmoid(logits)) - logits
        repellant_term = (
            -(1.0 - probabilities_graph)
            * (self.logsigmoid(log_probabilities_distance) - log_probabilities_distance)
            * Configs.negative_force
        )
        CE = attraction_term + repellant_term
        train_loss = CE.mean()

        losses = {"loss": train_loss}

        self.log(
            "train_loss",
            train_loss,
            batch_size=Configs.batch_size,
            **self.log_config_step,
        )

        save_folder = os.path.join(
            "./work_dirs",
            str(Configs.expr_index) + "_lr_" + str(Configs.lr),
        )
        os.makedirs(save_folder, exist_ok=True)
        if batch_idx % 250 == 0:
            # print(f"==>> i: {i}")
            # print("Ploting the figure")
            fig, ax = plt.subplots(figsize=(11.7, 8.27))
            with torch.no_grad():
                # X_code = self.model.code_space.detach().cpu().numpy()
                X_code = self.model(self.data_torch.to(self.device)).cpu().numpy()
                plt.scatter(
                    X_code[:, 0], X_code[:, 1], c=self.labels_numpy, s=5, cmap="tab10"
                )
                plt.savefig(
                    os.path.join(save_folder, "figure_" + str(batch_idx) + ".png")
                )
                plt.close(fig)

        return {"loss": losses["loss"]}

    # def training_epoch_end(self, outputs):
    #     cwd = os.getcwd()
    #     print("==>> Expriment Folder: ", cwd)

    #     embeddings = self.model.code_space.data.cpu().numpy()

    #     fig, ax = plt.subplots(figsize=(11.7, 8.27))
    #     plt.scatter(
    #         embeddings[:, 0],
    #         embeddings[:, 1],
    #         c=self.labels_numpy,
    #         s=5,
    #         cmap="tab10",
    #     )
    #     plt.savefig("./work_dirs/figure_cifar10_umap_pytorch.png")
    #     plt.close(fig)

    # train_accuracy = self.train_accuracy.compute()

    # """>>> If self.ignore is used, the last clss is fake and will not calculated in the metric; """
    # if self.ignore:
    #     train_accuracy_mean = torch.mean(train_accuracy[:-1]).item()
    # else:
    #     train_accuracy_mean = torch.mean(train_accuracy).item()

    # self.log(
    #     "train_accuracy_epoch",
    #     train_accuracy_mean,
    #     batch_size=Configs.batch_size,
    #     **self.log_config_epoch,
    # )

    # self.train_accuracy.reset()
    # self.train_precision.reset()
    # self.train_recall.reset()

    # def validation_step(self, batch, batch_idx):
    #     images, gt_labels = batch

    #     model_outputs = self.forward(images)
    #     val_loss = self.loss(model_outputs, gt_labels)
    #     # val_loss = val_losses_dict["loss"]

    #     model_predictions = model_outputs.argmax(dim=1)

    #     self.val_accuracy.update(model_predictions, gt_labels)
    #     self.val_precision.update(model_predictions, gt_labels)
    #     self.val_recall.update(model_predictions, gt_labels)

    #     self.log(
    #         "val_loss",
    #         val_loss,
    #         batch_size=Configs.batch_size * self.val_scales,
    #         **self.log_config_step,
    #     )

    #     return val_loss

    # def validation_epoch_end(self, outputs):
    #     """>>> The compute() function will return a list;"""
    #     val_epoch_accuracy = self.val_accuracy.compute()
    #     val_epoch_precision = self.val_precision.compute()
    #     val_epoch_recall = self.val_recall.compute()

    #     if self.ignore:
    #         val_epoch_accuracy_mean = torch.mean(val_epoch_accuracy[:-1].item())
    #         val_epoch_precision_mean = torch.mean(val_epoch_precision[:-1]).item()
    #         val_epoch_recall_mean = torch.mean(val_epoch_recall[:-1]).item()
    #     else:
    #         val_epoch_accuracy_mean = torch.mean(val_epoch_accuracy).item()
    #         val_epoch_precision_mean = torch.mean(val_epoch_precision).item()
    #         val_epoch_recall_mean = torch.mean(val_epoch_recall).item()

    #     self.log(
    #         "val_epoch_accuracy",
    #         val_epoch_accuracy_mean,
    #         batch_size=Configs.batch_size * self.val_scales,
    #         **self.log_config_epoch,
    #     )

    #     """>>> We use the "user_metric" variable to monitor the performance on val set; """
    #     user_metric = val_epoch_accuracy_mean
    #     self.log(
    #         "user_metric",
    #         user_metric,
    #         batch_size=Configs.batch_size * self.val_scales,
    #         **self.log_config_epoch,
    #     )

    #     """>>> Plot the results to console using only one gpu since the results on all gpus are the same; """
    #     if self.global_rank == 0:
    #         self.console_logger.info("epoch: {0:04d} ".format(self.current_epoch))

    #         for i in range(Configs.num_classes):
    #             if i < 20:
    #                 self.console_logger.info(
    #                     "{0: <15}, acc: {1:.4f} | precision: {2:.4f} | recall: {3:.4f}".format(
    #                         Configs.classes[i],
    #                         val_epoch_accuracy[i].item(),
    #                         val_epoch_precision[i].item(),
    #                         val_epoch_recall[i].item(),
    #                     )
    #                 )
    #         self.console_logger.info(
    #             "acc_mean: {0:.4f} ".format(val_epoch_accuracy_mean)
    #         )

    #         self.console_logger.info(
    #             "precision_mean: {0:.4f} ".format(val_epoch_precision_mean)
    #         )
    #         self.console_logger.info(
    #             "recall_mean: {0:.4f} ".format(val_epoch_recall_mean)
    #         )

    #     self.val_accuracy.reset()
    #     self.val_precision.reset()
    #     self.val_recall.reset()


class DataUtils:
    @staticmethod
    def get_weighted_knn(data, labels, metric="euclidean", n_neighbors=10):
        input_data_tensor = data.cuda()
        input_labels_tensor = labels.cuda()
        print("==>> input_data_tensor.shape: ", input_data_tensor.shape)
        print("==>> input_labels_tensor.shape: ", input_labels_tensor.shape)

        n_trees = 5 + int(round((data.shape[0]) ** 0.5 / 20.0))
        n_iters = max(5, int(round(np.log2(data.shape[0]))))
        # metric = "euclidean"
        # n_neighbors = 10
        nnd = NNDescent(
            data,
            n_neighbors=n_neighbors,
            metric=metric,
            n_trees=n_trees,
            n_iters=n_iters,
            max_candidates=60,
            verbose=True,
        )
        knn_index, knn_dist = nnd.neighbor_graph
        weighted_knn, _, _ = fuzzy_simplicial_set(
            data,
            n_neighbors=n_neighbors,
            random_state=None,
            metric="euclidean",
            metric_kwds={},
            knn_indices=knn_index,
            knn_dists=knn_dist,
            angular=False,
            set_op_mix_ratio=1.0,
            local_connectivity=1.0,
            apply_set_operations=True,
            verbose=False,
            return_dists=None,
        )
        print("weighted_knn: {}".format(weighted_knn.shape))

        return weighted_knn

    @staticmethod
    def get_graph_elements(graph_, n_epochs):
        """
        gets elements of graphs, weights, and number of epochs per edge

        Parameters
        ----------
        graph_ : scipy.sparse.csr.csr_matrix
            umap graph of probabilities
        n_epochs : int
            maximum number of epochs per edge

        Returns
        -------
        graph scipy.sparse.csr.csr_matrix
            umap graph
        epochs_per_sample np.array
            number of epochs to train each sample for
        head np.array
            edge head
        tail np.array
            edge tail
        weight np.array
            edge weight
        n_vertices int
            number of vertices in graph
        """
        ### should we remove redundancies () here??
        # graph_ = remove_redundant_edges(graph_)

        graph = graph_.tocoo()
        # eliminate duplicate entries by summing them together
        graph.sum_duplicates()
        # number of vertices in dataset
        n_vertices = graph.shape[1]
        # get the number of epochs based on the size of the dataset
        if n_epochs is None:
            # For smaller datasets we can use more epochs
            if graph.shape[0] <= 10000:
                n_epochs = 500
            else:
                n_epochs = 200
        # remove elements with very low probability
        graph.data[graph.data < (graph.data.max() / float(n_epochs))] = 0.0
        graph.eliminate_zeros()
        # get epochs per sample based upon edge probability
        epochs_per_sample = n_epochs * graph.data

        head = graph.row
        tail = graph.col
        weight = graph.data

        return graph, epochs_per_sample, head, tail, weight, n_vertices

    @staticmethod
    def get_edges_with_negative_sampling(weighted_knn):
        (
            graph,
            epochs_per_sample,
            head,
            tail,
            weight,
            n_vertices,
        ) = DataPreprocessing.get_graph_elements(weighted_knn, 200)
        print("==>> epochs_per_sample: ", epochs_per_sample)
        print("==>> epochs_per_sample.shape: ", epochs_per_sample.shape)
        print("==>> graph.shape: ", graph.shape)
        print("==>> head: ", head)
        print("==>> head.shape: ", head.shape)
        print("==>> tail: ", tail)
        print("==>> tail.shape: ", tail.shape)
        print("==>> weight: ", weight)
        print("==>> weight.shape: ", weight.shape)
        print("==>> n_vertices: ", n_vertices)
        # 这里是根据每个sample对一个的epochs_per_sample数目来复制node embedding, 也就是head和tail;
        # 这样我们就不需要再考虑weight的信息了;
        edges_to_exp, edges_from_exp = (
            np.repeat(head, epochs_per_sample.astype("int")),
            np.repeat(tail, epochs_per_sample.astype("int")),
        )
        shuffle_mask = np.random.permutation(range(len(edges_to_exp)))
        edges_to_exp = edges_to_exp[shuffle_mask].astype(np.int64)
        print("==>> edges_to_exp: ", edges_to_exp)
        print("==>> edges_to_exp.shape: ", edges_to_exp.shape)
        edges_from_exp = edges_from_exp[shuffle_mask].astype(np.int64)
        print("==>> edges_from_exp.shape: ", edges_from_exp.shape)

        return edges_to_exp, edges_from_exp, graph

    @staticmethod
    def get_umap_graph(X, n_neighbors=10, metric="cosine", random_state=None):
        random_state = (
            check_random_state(None) if random_state == None else random_state
        )
        # number of trees in random projection forest
        n_trees = 5 + int(round((X.shape[0]) ** 0.5 / 20.0))
        # max number of nearest neighbor iters to perform
        n_iters = max(5, int(round(np.log2(X.shape[0]))))
        # distance metric

        # get nearest neighbors
        nnd = NNDescent(
            X.reshape((len(X), np.product(np.shape(X)[1:]))),
            n_neighbors=n_neighbors,
            metric=metric,
            n_trees=n_trees,
            n_iters=n_iters,
            max_candidates=60,
            verbose=True,
        )
        # get indices and distances
        knn_indices, knn_dists = nnd.neighbor_graph

        # get indices and distances
        knn_indices, knn_dists = nnd.neighbor_graph
        # build fuzzy_simplicial_set
        umap_graph, sigmas, rhos = fuzzy_simplicial_set(
            X=X,
            n_neighbors=n_neighbors,
            metric=metric,
            random_state=random_state,
            knn_indices=knn_indices,
            knn_dists=knn_dists,
        )

        return umap_graph


class GetDINOEmbedding:
    @staticmethod
    def get_embedding_with_dinov2(dataloader, model, device="cpu"):
        import numpy as np
        import torch

        # vits16 = torch.hub.load("facebookresearch/dino:main", "dino_vits16")
        # # print("==>> vits16: ", vits16)
        # model = vits16
        model.eval()

        device = "cuda"

        output_dino_embeddings_list = []
        output_labels_list = []
        for images, labels in dataloader:
            with torch.no_grad():
                images = images.to(device)
                model = model.to(device)
                output = model(images)
                output_dino_embeddings_list.extend(output.cpu().numpy())
                output_labels_list.extend(labels.cpu().numpy())

        out_embeddings = np.array(output_dino_embeddings_list)
        out_labels = np.array(output_labels_list)

        return out_embeddings, out_labels


class GetKNNIndex:
    """+++ calculate the nearest neighbors of each input sample/image;"""

    @staticmethod
    def get_knn_index(images_numpy, topk, knn_filepath=None):
        import faiss  # not easy to install;
        import numpy as np

        images = images_numpy

        images_min = np.min(images)
        print("==>> images_min: ", images_min)
        images_max = np.max(images)
        print("==>> images_max: ", images_max)

        # Normalize the data

        features = images.reshape((images.shape[0], -1))
        n, dim = features.shape[0], features.shape[1]
        print(f"==>> n: {n}, dim: {dim}")

        index = faiss.IndexFlatIP(dim)
        index = faiss.index_cpu_to_all_gpus(index)
        index.add(features)

        """ +++ can also calculate both the index and the corresponding distance at the same time; """
        distances, indices = index.search(
            features, topk + 1
        )  # Sample itself is included

        if knn_filepath:
            np.savetxt(knn_filepath, indices, fmt="%d")

        return indices


class DataModule(pl.LightningDataModule):
    def __init__(self):
        super(DataModule, self).__init__()
        self.batch_size = Configs.batch_size

    def train_dataloader(self):
        train_loader = Data.get_dataloader(
            batch_size=self.batch_size,
            data_dir=Configs.data_dir,
            data_category="train",
        )

        return train_loader

    def val_dataloader(self):
        val_loader = Data.get_dataloader(
            batch_size=self.batch_size,
            data_dir=Configs.data_dir,
            data_category="val",
        )

        return val_loader


class Data:
    @staticmethod
    def get_dataloader(batch_size, data_dir, data_category="train"):
        if data_dir.split("/")[-1] == "cifar10":
            loader = Data.get_cifar10_dataloader(
                batch_size, data_dir, data_category=data_category
            )
        elif data_dir.split("/")[-1] == "imagenet":
            loader = Data.get_imagenet_dataloader(
                batch_size, data_dir, data_category=data_category
            )
        elif data_dir.split("/")[-1] == "umap_cifar10":
            loader, _, _ = Data.get_umap_edge_dataloader(
                batch_size, data_dir, data_category=data_category
            )
        else:
            print(f"Cannot recognize dataset.")
            import sys

            sys.exit()

        return loader

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
                transforms.Resize((Configs.image_raw_size, Configs.image_raw_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(Configs.image_crop_size, padding=0),
                transforms.ToTensor(),
                transforms.Normalize(Configs.mean, Configs.std),
            ]
        )

        dataset = torchvision.datasets.CIFAR10(
            root=data_dir,
            train=use_training,
            download=True,
            transform=transform,
        )

        if Configs.subtrain:
            sampling_pool = np.arange(len(dataset))
            np.random.shuffle(sampling_pool)
            num_sampling = int(Configs.subtrain_ratio * len(dataset))
            sublist = list(sampling_pool[:num_sampling])
            dataset = torch.utils.data.Subset(dataset, sublist)
            print("==>> sampled dataset: ", len(dataset))

        if Configs.subval:
            sampling_pool = np.arange(len(dataset))
            np.random.shuffle(sampling_pool)
            num_sampling = int(Configs.subval_ratio * len(dataset))
            sublist = list(sampling_pool[:num_sampling])
            dataset = torch.utils.data.Subset(dataset, sublist)
            print("==>> sampled dataset: ", len(dataset))

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=use_shuffle,
            num_workers=Configs.workers,
            pin_memory=Configs.pin_memory,
        )

        return loader

    @staticmethod
    def get_imagenet_dataloader(batch_size, data_dir, data_category="train"):
        import torch
        from torchvision import transforms
        from torchvision.datasets import ImageFolder

        if data_category == "train":
            use_training = True
            use_shuffle = True
        else:
            use_training = False
            use_shuffle = False

        # Define the transformations to be applied to the images
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),  # Resize the images to a fixed size
                transforms.ToTensor(),  # Convert images to tensors
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),  # Normalize the image tensors
            ]
        )

        # Create an instance of the ImageFolder dataset
        dataset = ImageFolder(
            root=os.path.join(data_dir, data_category), transform=transform
        )

        class_names = dataset.classes
        # print(f"==>> class_names: {class_names}")
        class_to_idx = dataset.class_to_idx
        # print(f"==>> class_to_idx: {class_to_idx}")

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=use_shuffle,
            num_workers=Configs.workers,
            pin_memory=Configs.pin_memory,
        )

        return loader

    @staticmethod
    def get_all_cifar10_data(resize=False):
        import torch
        import torchvision
        import torchvision.transforms as transforms

        if resize:
            transform = transforms.Compose(
                [
                    transforms.Resize((256, 256)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(224, padding=0),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            )

        dataset = torchvision.datasets.CIFAR10(
            root="./data/",
            train=False,
            download=True,
            transform=transform,
        )

        loader = torch.utils.data.DataLoader(
            dataset, batch_size=len(dataset), shuffle=False
        )

        images, labels = next(iter(loader))

        return images, labels

    @staticmethod
    def get_umap_edge_dataloader(batch_size, data_dir, data_category="train"):
        model = torch.hub.load("facebookresearch/dino:main", "dino_vits16")
        dataloader = Data.get_cifar10_dataloader(
            batch_size, data_dir="./data/", data_category="test"
        )
        out_embeddings, out_labels = GetDINOEmbedding.get_embedding_with_dinov2(
            dataloader, model, device="cuda"
        )
        out_embeddings = torch.from_numpy(out_embeddings)
        out_labels = torch.from_numpy(out_labels)
        print(f"==>> out_embeddings.shape: {out_embeddings.shape}")
        print(f"==>> out_labels.shape: {out_labels.shape}")

        images_torch = torch.tensor(out_embeddings)
        labels_torch = torch.tensor(out_labels)

        graph = DataUtils.get_umap_graph(
            images_torch.numpy(),
            n_neighbors=Configs.umap_n_neighbors,
            metric=Configs.umap_metric,
            random_state=Configs.umap_random_state,
        )
        umap_edge_dataset = UMAPDataset(out_embeddings, graph, n_epochs=10)

        loader = torch.utils.data.DataLoader(
            umap_edge_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=Configs.workers,
            pin_memory=Configs.pin_memory,
        )

        return loader, images_torch, labels_torch


class UMAPDataset(torch.utils.data.Dataset):
    def __init__(self, data, graph_, n_epochs=10):
        (
            graph,
            epochs_per_sample,
            head,
            tail,
            weight,
            n_vertices,
        ) = DataUtils.get_graph_elements(graph_, n_epochs)

        self.edges_to_exp, self.edges_from_exp = (
            np.repeat(head, epochs_per_sample.astype("int")),
            np.repeat(tail, epochs_per_sample.astype("int")),
        )
        shuffle_mask = np.random.permutation(np.arange(len(self.edges_to_exp)))
        self.edges_to_exp = self.edges_to_exp[shuffle_mask].astype(np.int64)
        self.edges_from_exp = self.edges_from_exp[shuffle_mask].astype(np.int64)
        self.data = torch.Tensor(data)

    def __len__(self):
        return int(self.edges_to_exp.shape[0])

    def __getitem__(self, index):
        edges_to_exp_with_data = self.data[self.edges_to_exp[index]]
        edges_from_exp_with_data = self.data[self.edges_from_exp[index]]
        return (edges_to_exp_with_data, edges_from_exp_with_data)


class Utils:
    @staticmethod
    def get_logger(name, log_file=None, log_level=logging.INFO, file_mode="w"):
        logger_initialized = {}

        logger = logging.getLogger(name)
        if name in logger_initialized:
            return logger
        for logger_name in logger_initialized:
            if name.startswith(logger_name):
                return logger
        for handler in logger.root.handlers:
            if type(handler) is logging.StreamHandler:
                handler.setLevel(logging.ERROR)

        stream_handler = logging.StreamHandler()
        handlers = [stream_handler]

        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
        else:
            rank = 0
        if rank == 0 and log_file is not None:
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

    @staticmethod
    def get_max_batchsize(data_dir, model):
        # Define the initial batch size and the maximum memory utilization
        max_memory_utilization = 0.9 * (
            torch.cuda.get_device_properties(0).total_memory / 1024**3
        )  # Use up to 90% of available GPU memory
        print(f"==>> max_memory_utilization: {max_memory_utilization}")

        # Initialize the batch size and memory usage
        batch_size = 1
        memory_usage = 0

        model.to("cuda")

        count = 0
        memeory_usage_dict = {}
        # Iterate until memory usage exceeds the threshold
        while memory_usage < max_memory_utilization:
            # print(f"==>> memory_usage: {memory_usage}")
            # Increase the batch size
            # if memory_usage < max_memory_utilization * 0.5:
            #     batch_size *= 2
            # else:
            #     batch_size += 1
            # print(f"==>> batch_size: {batch_size}")

            if count == 2:
                memory_ratio_int = int(
                    (max_memory_utilization - memeory_usage_dict[0])
                    // (memeory_usage_dict[1] - memeory_usage_dict[0])
                )
                batch_size *= memory_ratio_int
                break

            # elif count > 1:
            #     batch_size += 1

            batch_size = int(batch_size)
            print(f"==>> batch_size: {batch_size}")

            dataloader = Data.get_dataloader(
                batch_size, data_dir, data_category="train"
            )

            # Iterate over a small number of batches to measure memory usage
            for images, labels in dataloader:
                images = images.to("cuda")
                labels = labels.to("cuda")
                _ = model(images)

                memory_usage = torch.cuda.max_memory_allocated() / 1024**3
                print(f"==>> memory_usage: {memory_usage}")

                memeory_usage_dict[count] = memory_usage

                images = images.to("cpu")
                labels = labels.to("cpu")
                del images
                del labels
                break
            count += 1
        del model

        print(f"Selected batch size: {batch_size}")

        return batch_size


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
    lr = 0.08
    opt = "SGD"
    backbone_lr = 0.01

    ### Training Config:
    subtrain = False
    subtrain_ratio = 0.1

    batch_size = 16 * 8 * 2
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
    num_gpus = 1  # "autocount"
    num_nodes = 1
    precision = 16
    strategy = "ddp"
    accelerator = "gpu"
    progress_bar_refresh_rate = 1

    T_max = 100  # for cosineAnn; The same with max epoch
    eta_min = 1e-6  # for cosineAnn
    T_0 = 5  # for cosineAnnWarm; cosineAnnWarmDecay
    T_mult = 9  # for cosineAnnWarm; cosineAnnWarmDecay
    max_epochs = T_0 + T_mult * T_0

    ### Validation Config:
    subval = False
    subval_ratio = 0.1
    val_interval = 1
    val_batch_size = 16

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


if __name__ == "__main__":
    SettingUtils.generate_folders()

    loader, images_torch, labels_torch = Data.get_umap_edge_dataloader(
        Configs.batch_size, data_dir="./data"
    )

    model = Models(images_torch, labels_torch)
    num_gpus = Settings.gpu_setting()
    pl_trainer_for_train = Settings.pl_trainer_setting(num_gpus=num_gpus)
    data_module = Settings.data_module_setting()

    if Configs.training_mode:
        pl_trainer_for_train.fit(
            model,
            data_module,
        )
    else:
        pl_trainer_for_train.validate(model, data_module)
