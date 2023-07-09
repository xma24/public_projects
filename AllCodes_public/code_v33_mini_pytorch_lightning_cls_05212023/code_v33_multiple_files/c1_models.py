import os

import pytorch_lightning as pl
import timm
import torch
import torch.distributed as dist
import torch.nn as nn
from c3_utils import Utils
from c5_configs import Configs
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR


class ModelUtils:
    @staticmethod
    def get_efficientb0_from_timm(num_classes):
        import timm

        keyworlds = "efficientnet"
        selected_listed_models = ModelUtils.get_timm_model_list(keyworlds)
        # print(f"==>> selected_listed_models: {selected_listed_models}")

        model_name = "efficientnet_b0.ra_in1k"
        init_model = timm.create_model(model_name, pretrained=False)
        feature_dim = init_model.get_classifier().in_features

        backbone = nn.Sequential()
        init_classifier = nn.Sequential()
        named_modules_list = list(init_model.named_children())

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
        # print("==>> listed_models len: ", len(listed_models))

        listed_pretraiend_models = timm.list_models(pretrained=True)
        # print("==>> listed_pretraiend_models: ", listed_pretraiend_models)
        # print("==>> listed_pretraiend_models len: ", len(listed_pretraiend_models))

        selected_listed_models = timm.list_models("*" + keyworlds + "*")
        # print("==>> selected_listed_models: ", selected_listed_models)
        # print("==>> selected_listed_models len: ", len(selected_listed_models))

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
    def load_pretrained_weights(model):
        checkpoint = torch.load(Configs.pretrained_weights)
        state_dict = checkpoint["state_dict"]
        modified_state_dict = {}
        for key, value in state_dict.items():
            modified_key = key[len("model.") :]
            modified_state_dict[modified_key] = value
        ret_loading = model.load_state_dict(modified_state_dict, strict=True)
        print(f"\n ==>> ret_loading: {ret_loading} \n")

        return model


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
        ) = PytorchLightningDefault.get_classification_metrics(
            Configs.num_classes, self.ignore
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


class Models(PytorchLightningDefault):
    def __init__(self, logger=None):
        super(Models, self).__init__()

        self.module_lr_dict = dict(placeholder=0.0)

        (
            init_model,
            backbone,
            init_classifier,
            classifier,
        ) = ModelUtils.get_efficientb0_from_timm(num_classes=Configs.num_classes)
        # ModelUtils.check_weights_loading(init_model)

        # init_model_bk = nn.Sequential()
        # init_model_bk.add_module("backbone", backbone)
        # init_model_bk.add_module("init_classifier", init_classifier)
        # # init_model_bk = init_model_bk.to(Configs.device)

        # ModelUtils.check_weights_loading(init_model_bk)

        self.model = nn.Sequential()
        self.model.add_module("backbone", backbone)
        self.model.add_module("classifier", classifier)
        # model = model.to(Configs.device)


        if Configs.pretrained_weights:
            ModelUtils.load_pretrained_weights(self.model)

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
            batch_size=Configs.batch_size,
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
            batch_size=Configs.batch_size,
            **self.log_config_epoch,
        )

        self.train_accuracy.reset()
        self.train_precision.reset()
        self.train_recall.reset()

    def validation_step(self, batch, batch_idx):
        images, gt_labels = batch

        model_outputs = self.forward(images)
        print(f"==>> model_outputs.shape: {model_outputs.shape}")

        gathered_outputs = [
            torch.zeros_like(model_outputs) for _ in range(dist.get_world_size())
        ]
        dist.all_gather(gathered_outputs, model_outputs)
        merged_outputs = torch.cat(gathered_outputs, dim=0)
        print(f"==>> merged_outputs.shape: {merged_outputs.shape}")

        val_loss = self.loss(model_outputs, gt_labels)
        # val_loss = val_losses_dict["loss"]

        model_predictions = model_outputs.argmax(dim=1)

        self.val_accuracy.update(model_predictions, gt_labels)
        self.val_precision.update(model_predictions, gt_labels)
        self.val_recall.update(model_predictions, gt_labels)

        self.log(
            "val_loss",
            val_loss,
            batch_size=Configs.batch_size * self.val_scales,
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
            batch_size=Configs.batch_size * self.val_scales,
            **self.log_config_epoch,
        )

        """>>> We use the "user_metric" variable to monitor the performance on val set; """
        user_metric = val_epoch_accuracy_mean
        self.log(
            "user_metric",
            user_metric,
            batch_size=Configs.batch_size * self.val_scales,
            **self.log_config_epoch,
        )

        """>>> Plot the results to console using only one gpu since the results on all gpus are the same; """
        if self.global_rank == 0:
            self.console_logger.info("epoch: {0:04d} ".format(self.current_epoch))

            for i in range(Configs.num_classes):
                self.console_logger.info(
                    "{0: <15}, acc: {1:.4f} | precision: {2:.4f} | recall: {3:.4f}".format(
                        Configs.classes[i],
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
