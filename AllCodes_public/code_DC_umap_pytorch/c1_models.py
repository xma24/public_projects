import os

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import timm
import torch
import torch.nn as nn
import torchmetrics
from c5_configs import Configs
from sklearn.neighbors import KDTree
from sklearn.utils import check_random_state
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from umap.spectral import spectral_layout
from c3_utils import Utils


class ModelUtils(pl.LightningModule):
    """_summary_

    Args:
        pl (_type_): _description_
    """

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
        """_summary_

        Args:
            num_classes (_type_): _description_
            ignore (_type_): _description_

        Returns:
            _type_: _description_
        """
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

        for param_group_idx, sub_param_group in enumerate(param_groups):
            sub_lr_name = "lr/lr_" + str(param_group_idx)
            self.log(
                sub_lr_name,
                sub_param_group["lr"],
                batch_size=Configs.batch_size,
                **self.log_config_step,
            )

    def configure_optimizers(self):
        """_summary_

        Returns:
            _type_: _description_
        """
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
        """_summary_

        Args:
            module_lr_dict (_type_): _description_
            lr (_type_): _description_
        """

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
        """_summary_

        Raises:
            NotImplementedError: _description_

        Returns:
            _type_: _description_
        """
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
        """_summary_

        Args:
            num_classes (_type_): _description_

        Returns:
            _type_: _description_
        """
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
        """_summary_

        Returns:
            _type_: _description_
        """
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
        """_summary_

        Args:
            num_classes (_type_): _description_

        Returns:
            _type_: _description_
        """
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
    def get_resnet18_timm(num_classes):
        """_summary_

        Returns:
            _type_: _description_
        """
        import timm

        # keyworlds = "resnet"
        # selected_listed_models = ModelUtils.get_timm_model_list(keyworlds)
        # print(f"==>> selected_listed_models: {selected_listed_models}")
        # import sys
        # sys.exit()

        model_name = "resnet18"  # "vit_small_patch32_224", "resnet18"
        init_model = timm.create_model(model_name, pretrained=False)
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
    def get_mlp(num_classes):
        """_summary_

        Args:
            num_classes (_type_): _description_

        Returns:
            _type_: _description_
        """
        fc1 = nn.Linear(384, 128)
        fc2 = nn.Linear(128, 256)
        fc3 = nn.Linear(256, 512)
        fc4 = nn.Linear(512, 256)
        fc5 = nn.Linear(256, num_classes)
        relu = nn.ReLU()

        encoder = nn.Sequential()
        encoder.add_module("fc1", fc1)
        encoder.add_module("relu", relu)
        encoder.add_module("fc2", fc2)
        encoder.add_module("relu", relu)
        encoder.add_module("fc3", fc3)
        encoder.add_module("relu", relu)
        encoder.add_module("fc4", fc4)
        encoder.add_module("relu", relu)
        encoder.add_module("fc5", fc5)

        return encoder

    @staticmethod
    def get_timm_model_list(keyworlds):
        """_summary_

        Args:
            keyworlds (_type_): _description_

        Returns:
            _type_: _description_
        """
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

    # @staticmethod
    # def get_clustering_encoder(data, graph):
    #     """_summary_

    #     Args:
    #         data (_type_): _description_
    #         graph (_type_): _description_

    #     Returns:
    #         _type_: _description_
    #     """
    #     model = CodeSpaceTable(
    #         continuous=Configs.continuous,
    #         n_points=Configs.num_samples,
    #         dim=Configs.dim,
    #     ).cuda()

    #     init_embedding = ModelUtils.init_embedding_from_graph(
    #         data.numpy(),
    #         graph,
    #         Configs.dim,
    #         random_state=None,
    #         metric="euclidean",
    #         _metric_kwds={},
    #         init="spectral",
    #     )

    #     model.code_space.data = torch.from_numpy(init_embedding).cuda()

    #     return model

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


class Models(ModelUtils):
    """_summary_

    Args:
        ModelUtils (_type_): _description_
    """

    def __init__(self, images_torch, labels_torch, logger=None):
        super(Models, self).__init__()

        self.module_lr_dict = dict(placeholder=0.0)

        self.val_embeddings_list = []
        self.val_lables_list = []

        # (
        #     init_model,
        #     backbone,
        #     init_classifier,
        #     classifier,
        # ) = ModelUtils.get_resnet18_timm(num_classes=Configs.umap_n_components)

        # self.model = nn.Sequential()
        # self.model.add_module("backbone", backbone)
        # self.model.add_module("classifier", classifier)

        self.images_torch = images_torch
        self.labels_torch = labels_torch

        # self.model_dino = torch.hub.load(
        #     "facebookresearch/dino:main", "dino_vits16"
        # ).to(self.device)

        self.model = ModelUtils.get_mlp(num_classes=Configs.umap_n_components)

        self.logsigmoid = torch.nn.LogSigmoid()

        self.batch_size = Configs.batch_size
        self.save_hyperparameters()

    def forward(self, images):
        """_summary_

        Args:
            images (_type_): _description_

        Returns:
            _type_: _description_
        """
        embeddings = self.model(images)
        return embeddings

    def training_step(self, batch, batch_idx):
        """_summary_

        Args:
            batch (_type_): _description_
            batch_idx (_type_): _description_

        Returns:
            _type_: _description_
        """
        self.lr_logging()

        edge_to_exp_with_data, edge_from_exp_with_data = batch
        edge_to_exp_embeddings = self.model(edge_to_exp_with_data)
        edge_from_exp_embeddings = self.model(edge_from_exp_with_data)
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
            "val_loss",
            train_loss,
            batch_size=Configs.batch_size,
            **self.log_config_step,
        )

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
        if batch_idx % 500 == 0:
            # print(f"==>> i: {i}")
            # print("Ploting the figure")
            fig, ax = plt.subplots(figsize=(11.7, 8.27))
            with torch.no_grad():
                # X_code = self.model.code_space.detach().cpu().numpy()
                X_code = self.model(self.images_torch.to(self.device)).cpu().numpy()
                plt.scatter(
                    X_code[:, 0],
                    X_code[:, 1],
                    c=self.labels_torch.cpu().numpy(),
                    s=5,
                    cmap="tab10",
                )
                plt.savefig(
                    os.path.join(
                        save_folder,
                        "figure_e"
                        + str(self.current_epoch)
                        + "_bs_"
                        + str(batch_idx)
                        + ".jpg",
                    )
                )
                plt.close(fig)

        return {"loss": losses["loss"]}

    # def validation_step(self, batch, batch_idx):
    #     if batch_idx == 0:
    #         self.val_embeddings_list = []
    #         self.val_lables_list = []
    #     images, gt_labels = batch
    #     print(f"==>> batch_idx: {batch_idx}, images: {images.shape}")

    #     embeddings_dino = self.model_dino(images)
    #     embedings = self.forward(embeddings_dino)
    #     print(f"==>> embedings.shape: {embedings.shape}")

    #     self.val_embeddings_list.extend(embedings)
    #     self.val_lables_list.extend(gt_labels)

    # def validation_epoch_end(self, outputs):
    #     """_summary_

    #     Args:
    #         outputs (_type_): _description_
    #     """
    #     save_folder = os.path.join(
    #         "./work_dirs",
    #         str(Configs.expr_index) + "_lr_" + str(Configs.lr),
    #     )
    #     os.makedirs(save_folder, exist_ok=True)

    #     fig, _ = plt.subplots(figsize=(11.7, 8.27))
    #     with torch.no_grad():
    #         # X_code = self.model.code_space.detach().cpu().numpy()

    #         X_code = torch.stack(self.val_embeddings_list).cpu().numpy()
    #         print(f"==>> X_code.shape: {X_code.shape}")
    #         labels = torch.stack(self.val_lables_list).cpu().numpy()
    #         print(f"==>> labels.shape: {labels.shape}")
    #         plt.scatter(X_code[:, 0], X_code[:, 1], c=labels, s=5, cmap="tab10")
    #         plt.savefig(
    #             os.path.join(save_folder, "figure_" + str(self.current_epoch) + ".png")
    #         )
    #         plt.close(fig)
