import torch


class ExprCommonSetting:
    def generate_folders():

        import os

        data_folder = "./data/"
        work_dirs_folder = "./work_dirs/"

        os.makedirs(data_folder, exist_ok=True)
        os.makedirs(work_dirs_folder, exist_ok=True)


class GetDataloaderCIFAR10:
    @staticmethod
    def get_cifar10_dataloader(batch_size, data_category="train"):
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
                transforms.Resize((256, 256)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(224, padding=0),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        dataset = torchvision.datasets.CIFAR10(
            root="/data/SSD1/data/cifar10_online/",
            train=use_training,
            download=True,
            transform=transform,
        )

        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=use_shuffle
        )

        return loader


class GetDINOEmbedding:
    @staticmethod
    def get_embedding_with_dinov2(dataloader, model, device="cpu"):
        import torch
        import numpy as np

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


class GetTSNEEmbedding:
    """+++ use tsne;"""

    @staticmethod
    def tsne_embedding_plot(
        images, labels, use_pca=False, tsne_image_path="./test_image.png"
    ):

        import numpy as np
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt

        data = images.reshape((images.shape[0], -1))
        labels = labels

        if use_pca:
            from sklearn.decomposition import PCA

            pca = PCA(n_components=32)
            data = pca.fit_transform(data)

        print("Start using TSNE ...")
        tsne = TSNE(n_components=2, random_state=42)
        tsne_embedded = tsne.fit_transform(data)

        plt.scatter(tsne_embedded[:, 0], tsne_embedded[:, 1], c=labels, s=3, cmap="jet")
        plt.colorbar()
        plt.savefig(tsne_image_path)
        plt.close()


class GetUMAPEmbedding:
    @staticmethod
    def umap_embedding_plot(images, labels, umap_image_path="./test_image.png"):
        import umap.umap_ as umap
        import matplotlib.pyplot as plt

        data = images.numpy().reshape((images.shape[0], -1))
        labels = labels.numpy()

        umap_embeddings = umap.UMAP(random_state=42).fit_transform(data)

        plt.scatter(
            umap_embeddings[:, 0], umap_embeddings[:, 1], c=labels, s=3, cmap="jet"
        )
        plt.colorbar()
        plt.savefig(umap_image_path)


class GetUMAPthenTSNEEmbedding:
    @staticmethod
    def umap_tsne_embedding_plot(
        images, labels, use_pca=False, out_image_path="./test_image.png"
    ):

        import umap.umap_ as umap
        import matplotlib.pyplot as plt

        data = images.reshape((images.shape[0], -1))
        labels = labels

        umap_embeddings = umap.UMAP(n_components=100, random_state=42).fit_transform(
            data
        )

        GetTSNEEmbedding.tsne_embedding_plot(
            umap_embeddings, labels, use_pca=use_pca, tsne_image_path=out_image_path
        )


if __name__ == "__main__":
    batch_size = 32
    model = torch.hub.load("facebookresearch/dino:main", "dino_vits16")

    ExprCommonSetting.generate_folders()

    """ +++ get the embedding of input dataset; """
    dataloader = GetDataloaderCIFAR10.get_cifar10_dataloader(
        batch_size, data_category="test"
    )
    out_embeddings, out_labels = GetDINOEmbedding.get_embedding_with_dinov2(
        dataloader, model, device="cuda"
    )
    print(f"==>> out_embeddings.shape: {out_embeddings.shape}")
    print(f"==>> out_labels.shape: {out_labels.shape}")

    """ +++ use tsne without pca; """
    # use_pca = False
    # save_output_path = "./work_dirs/code_v6_dinov2_cifar10_tsne_wo_pca.png"
    # GetTSNEEmbedding.tsne_embedding_plot(
    #     out_embeddings, out_labels, use_pca=use_pca, tsne_image_path=save_output_path
    # )

    """ +++ use tsne with pca; """
    # use_pca = True
    # save_output_path = "./work_dirs/code_v6_dinov2_cifar10_tsne_w_pca.png"
    # GetTSNEEmbedding.tsne_embedding_plot(
    #     out_embeddings, out_labels, use_pca=use_pca, tsne_image_path=save_output_path
    # )

    """ +++ use umap first then tsne; """
    save_output_path = "./work_dirs/code_v6_dinov2_cifar10_umap_then_tsne_wo_pca.png"
    GetUMAPthenTSNEEmbedding.umap_tsne_embedding_plot(
        out_embeddings, out_labels, use_pca=False, out_image_path=save_output_path
    )
