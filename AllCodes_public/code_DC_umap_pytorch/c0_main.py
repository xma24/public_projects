from c5_configs import Configs
from c1_models import Models
from c4_settings import (
    Settings,
    SettingUtils,
)
from c2_data import Data

if __name__ == "__main__":
    SettingUtils.generate_folders()

    _, images_torch, labels_torch = Data.get_umap_edge_dataloader(
        Configs.batch_size, "./data", data_category="train"
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
