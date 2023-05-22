import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
import timm


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
            "dog.jpg",
        )
        try:
            urllib.URLopener().retrieve(url, filename)
        except:
            urllib.request.urlretrieve(url, filename)

        url_imagenet_classes, filename_imagenet_classes = (
            "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt",
            "imagenet_classes.txt",
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
        with open("imagenet_classes.txt", "r") as f:
            categories = [s.strip() for s in f.readlines()]

        print(f"\n prediction: \n")
        top5_prob, top5_catid = torch.topk(probabilities, 5)
        for i in range(top5_prob.size(0)):
            print(categories[top5_catid[i]], top5_prob[i].item())


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
                transforms.Resize((250, 250)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(224, padding=0),
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
            dataset, batch_size=batch_size, shuffle=use_shuffle
        )

        return loader


class PytorchRun:
    @staticmethod
    def train_model(
        model, train_loader, test_loader, num_epochs, print_interval=10, device="cpu"
    ):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        eta_min = TrainingConfig.eta_min
        T_max = num_epochs
        last_epoch = -1
        scheduler = CosineAnnealingLR(
            optimizer, T_max=T_max, eta_min=eta_min, last_epoch=last_epoch
        )

        for epoch in range(num_epochs):
            for batch_idx, (images, labels) in enumerate(train_loader):
                pre_time = time.time()
                if torch.cuda.is_available():
                    images = images.to(device)
                    labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                current_lr = optimizer.param_groups[0]["lr"]
                diff_time = time.time() - pre_time
                if batch_idx % print_interval == 0:
                    avg_loss, accuracy = PytorchRun.evaluate_model(
                        model,
                        test_loader,
                        criterion,
                        device,
                    )
                    print(
                        f"Epoch [{epoch+1}/{num_epochs}], current_lr: {current_lr}, batch_time: {diff_time}, Batch [{batch_idx+1}/{len(train_loader)}], Train Loss: {loss.item()}, Val Loss: {avg_loss}, Val Acc: {accuracy}"
                    )
        torch.save(model.state_dict(), "./work_dirs/model.pth")

    @staticmethod
    def evaluate_model(model, dataloader, loss_fn, device):
        model.eval()  # Set the model to evaluation mode
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                total_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total
        model.train()

        return avg_loss, accuracy


class ExprCommonSetting:
    @staticmethod
    def generate_folders():
        import os

        data_folder = "./data/"
        work_dirs_folder = "./work_dirs/"

        os.makedirs(data_folder, exist_ok=True)
        os.makedirs(work_dirs_folder, exist_ok=True)


class TrainingConfig:
    batch_size = 32 * 8
    data_dir = "/data/SSD1/data/cifar10_online/"
    # data_category = "train"
    num_epochs = 10
    print_interval = 20
    num_classes = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eta_min = 1e-6


if __name__ == "__main__":

    ExprCommonSetting.generate_folders()

    train_loader = GetImagesCIFAR10.get_cifar10_dataloader(
        batch_size=TrainingConfig.batch_size,
        data_dir=TrainingConfig.data_dir,
        data_category="train",
    )

    test_loader = GetImagesCIFAR10.get_cifar10_dataloader(
        batch_size=TrainingConfig.batch_size,
        data_dir=TrainingConfig.data_dir,
        data_category="test",
    )

    init_model, backbone, init_classifier, classifier = TorchModel.get_model(
        num_classes=TrainingConfig.num_classes
    )
    TorchModel.check_weights_loading(init_model)

    init_model_bk = nn.Sequential()
    init_model_bk.add_module("backbone", backbone)
    init_model_bk.add_module("init_classifier", init_classifier)
    init_model_bk = init_model_bk.to(TrainingConfig.device)

    TorchModel.check_weights_loading(init_model_bk)

    model = nn.Sequential()
    model.add_module("backbone", backbone)
    model.add_module("classifier", classifier)
    model = model.to(TrainingConfig.device)

    PytorchRun.train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=TrainingConfig.num_epochs,
        print_interval=TrainingConfig.print_interval,
        device=TrainingConfig.device,
    )

    # Step 6: Evaluation (optional)
    # After training, you can evaluate the model on a validation set or perform other tasks

    # Step 7: Save the model (optional)
    # You can save the trained model for future use
    # torch.save(model.state_dict(), "./work_dirs/model.pth")
