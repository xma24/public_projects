import torch
from c1_models import Models, ModelUtils
from c2_data import Data, DataModule
from c4_settings import Settings, SettingUtils
from c5_configs import Configs


class SKModel:
    def __init__(
        self,
    ):
        super().__init__()

        self.data_module = DataModule()
        self.model = Models()

    def fit(self):
        pl_trainer = Settings.pl_trainer_setting()
        if Configs.training_mode:
            pl_trainer.fit(self.model, self.data_module)
        else:
            pl_trainer.validate(self.model, self.data_module)
            # pl_trainer.test(model, dataloaders=val_dataloader)

    @torch.no_grad()
    def transform(self, X):
        return self.model(X).detach().cpu().numpy()


if __name__ == "__main__":
    SettingUtils.generate_folders()

    """ +++ check model loading process. make sure it is correct."""
    # init_model, backbone, init_classifier, classifier = ModelUtils.get_model(
    #     num_classes=DataConfig.num_classes
    # )
    # ModelUtils.check_weights_loading(init_model)

    # init_model_bk = nn.Sequential()
    # init_model_bk.add_module("backbone", backbone)
    # init_model_bk.add_module("init_classifier", init_classifier)
    # # init_model_bk = init_model_bk.to(Configs.device)

    # ModelUtils.check_weights_loading(init_model_bk)

    # data_module = DataModule()
    # model = Models()
    # pl_trainer = Settings.pl_trainer_setting()

    # if Configs.training_mode:
    #     pl_trainer.fit(model, data_module)
    # else:
    #     pl_trainer.validate(model, data_module)
    #     # pl_trainer.test(model, dataloaders=val_dataloader)

    print(f"\nParameters used:")
    for key, value in Configs.__dict__.items():
        if not key.startswith("__"):
            print(f"{key}: {value}")
    print(f"\n")

    skmodel = SKModel()
    skmodel.fit()

    # val_dataloader = Data.get_cifar10_dataloader(
    #     batch_size=16, data_dir="./data", data_category="val"
    # )

    val_dataloader = skmodel.data_module.val_dataloader()

    for images, labels in val_dataloader:
        images = images
        print(f"==>> images: {images.device}")
        labels = labels
        output = skmodel.transform(images)
        # print(f"==>> output: {output}")
        print(f"==>> output.shape: {output.shape}")

        break
