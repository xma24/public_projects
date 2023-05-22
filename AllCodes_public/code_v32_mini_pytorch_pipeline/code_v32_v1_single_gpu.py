import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import time


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc1 = nn.Linear(32, 10)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.pool1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.pool2(out)
        out = self.adaptive_avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        return out


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
                transforms.Resize((40, 40)),
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
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

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
                # scheduler.step()
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    model = CNN().to(TrainingConfig.device)
    # device_ids = list(range(torch.cuda.device_count()))
    # print(f"==>> device_ids: {device_ids}")

    # model = nn.DataParallel(model, device_ids=device_ids)

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
