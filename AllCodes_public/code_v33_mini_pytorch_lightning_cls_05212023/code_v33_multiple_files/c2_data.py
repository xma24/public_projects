import pytorch_lightning as pl
import torch
import torchvision
import torchvision.transforms as transforms
from c5_configs import Configs
import albumentations as A
from albumentations.pytorch import ToTensorV2


class DataUtils:
    @staticmethod
    def placeholder():
        pass

    @staticmethod
    def classification_augmentation_A():
        transform = A.Compose(
            [
                A.Resize(height=256, width=256),
                A.RandomCrop(height=224, width=224),
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=15),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
                A.GaussianBlur(blur_limit=3),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15),
                A.RandomResizedCrop(
                    height=224, width=224, scale=(0.8, 1.0), ratio=(0.9, 1.1)
                ),
                A.Normalize(),
                ToTensorV2(),
            ]
        )
        return transform


class DataModule(pl.LightningDataModule):
    """_summary_

    Args:
        pl (_type_): _description_
    """

    def __init__(self):
        super(DataModule, self).__init__()
        self.batch_size = Configs.batch_size

    def train_dataloader(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        train_loader = Data.get_dataloader(
            batch_size=self.batch_size,
            data_dir=Configs.data_dir,
            data_category="train",
        )

        return train_loader

    def val_dataloader(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        val_loader = Data.get_dataloader(
            batch_size=self.batch_size,
            data_dir=Configs.data_dir,
            data_category="val",
        )

        return val_loader


class AugmentedCIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, root, train, download, transform):
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label


class Data:
    @staticmethod
    def get_dataloader(batch_size, data_dir, data_category="train"):
        """_summary_

        Args:
            batch_size (_type_): _description_
            data_dir (_type_): _description_
            data_category (str, optional): _description_. Defaults to "train".

        Returns:
            _type_: _description_
        """
        if data_dir.split("/")[-1] == "cifar10":
            loader = Data.get_cifar10_dataloader(
                batch_size, data_dir, data_category=data_category
            )
        else:
            print(f"Cannot recognize dataset.")
            import sys

            sys.exit()

        return loader

    @staticmethod
    def get_cifar10_dataloader(batch_size, data_dir, data_category="train"):
        if data_category == "train":
            use_training = True
            use_shuffle = True
        else:
            use_training = False
            use_shuffle = False

        transform = DataUtils.classification_augmentation_A()

        dataset = AugmentedCIFAR10(
            root=data_dir,
            train=use_training,
            download=True,
            transform=transform,
        )

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=use_shuffle,
            num_workers=Configs.workers,
            pin_memory=Configs.pin_memory,
        )

        return loader
