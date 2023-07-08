from c1_models import Models
from c4_settings import Settings, SettingUtils
from c5_configs import Configs
from c2_data import DataModule
from c1_models import ModelUtils

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

    data_module = DataModule()
    model = Models()
    pl_trainer = Settings.pl_trainer_setting()

    if Configs.training_mode:
        pl_trainer.fit(model, data_module)
    else:
        pl_trainer.validate(model, data_module)
        # pl_trainer.test(model, dataloaders=val_dataloader)
