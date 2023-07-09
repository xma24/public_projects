import os

import numpy as np
import pytorch_lightning as pl
import torch
from c5_configs import Configs
from pynndescent import NNDescent
from sklearn.utils import check_random_state
from umap.umap_ import fuzzy_simplicial_set


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

    # @staticmethod
    # def get_edges_with_negative_sampling(weighted_knn):
    #     (
    #         graph,
    #         epochs_per_sample,
    #         head,
    #         tail,
    #         weight,
    #         n_vertices,
    #     ) = DataPreprocessing.get_graph_elements(weighted_knn, 200)
    #     print("==>> epochs_per_sample: ", epochs_per_sample)
    #     print("==>> epochs_per_sample.shape: ", epochs_per_sample.shape)
    #     print("==>> graph.shape: ", graph.shape)
    #     print("==>> head: ", head)
    #     print("==>> head.shape: ", head.shape)
    #     print("==>> tail: ", tail)
    #     print("==>> tail.shape: ", tail.shape)
    #     print("==>> weight: ", weight)
    #     print("==>> weight.shape: ", weight.shape)
    #     print("==>> n_vertices: ", n_vertices)
    #     # 这里是根据每个sample对一个的epochs_per_sample数目来复制node embedding, 也就是head和tail;
    #     # 这样我们就不需要再考虑weight的信息了;
    #     edges_to_exp, edges_from_exp = (
    #         np.repeat(head, epochs_per_sample.astype("int")),
    #         np.repeat(tail, epochs_per_sample.astype("int")),
    #     )
    #     shuffle_mask = np.random.permutation(range(len(edges_to_exp)))
    #     edges_to_exp = edges_to_exp[shuffle_mask].astype(np.int64)
    #     print("==>> edges_to_exp: ", edges_to_exp)
    #     print("==>> edges_to_exp.shape: ", edges_to_exp.shape)
    #     edges_from_exp = edges_from_exp[shuffle_mask].astype(np.int64)
    #     print("==>> edges_from_exp.shape: ", edges_from_exp.shape)

    #     return edges_to_exp, edges_from_exp, graph

    @staticmethod
    def get_umap_graph(X, n_neighbors=10, metric="cosine", random_state=None):
        """_summary_

        Args:
            X (_type_): _description_
            n_neighbors (int, optional): _description_. Defaults to 10.
            metric (str, optional): _description_. Defaults to "cosine".
            random_state (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
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
    """_summary_

    Returns:
        _type_: _description_
    """

    @staticmethod
    def get_embedding_with_dinov2(dataloader, model, device="cpu"):
        """_summary_

        Args:
            dataloader (_type_): _description_
            model (_type_): _description_
            device (str, optional): _description_. Defaults to "cpu".

        Returns:
            _type_: _description_
        """
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


# class GetKNNIndex:
#     """+++ calculate the nearest neighbors of each input sample/image;"""

#     @staticmethod
#     def get_knn_index(images_numpy, topk, knn_filepath=None):
#         import faiss  # not easy to install;
#         import numpy as np

#         images = images_numpy

#         images_min = np.min(images)
#         print("==>> images_min: ", images_min)
#         images_max = np.max(images)
#         print("==>> images_max: ", images_max)

#         # Normalize the data

#         features = images.reshape((images.shape[0], -1))
#         n, dim = features.shape[0], features.shape[1]
#         print(f"==>> n: {n}, dim: {dim}")

#         index = faiss.IndexFlatIP(dim)
#         index = faiss.index_cpu_to_all_gpus(index)
#         index.add(features)

#         """ +++ can also calculate both the index and the corresponding distance at the same time; """
#         distances, indices = index.search(
#             features, topk + 1
#         )  # Sample itself is included

#         if knn_filepath:
#             np.savetxt(knn_filepath, indices, fmt="%d")

#         return indices


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


class UMAPDataset(torch.utils.data.Dataset):
    """_summary_

    Args:
        torch (_type_): _description_
    """

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
        self.data = data
        print(f"==>> self.data: {self.data.shape}")

    def __len__(self):
        return int(self.edges_to_exp.shape[0])

    def __getitem__(self, index):
        edges_to_exp_with_data = self.data[self.edges_to_exp[index]]
        edges_from_exp_with_data = self.data[self.edges_from_exp[index]]
        return (edges_to_exp_with_data, edges_from_exp_with_data)


class Data:
    """_summary_

    Returns:
        _type_: _description_
    """

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
        # elif data_dir.split("/")[-1] == "imagenet":
        #     loader = Data.get_imagenet_dataloader(
        #         batch_size, data_dir, data_category=data_category
        #     )
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
        """_summary_

        Args:
            batch_size (_type_): _description_
            data_dir (_type_): _description_
            data_category (str, optional): _description_. Defaults to "train".

        Returns:
            _type_: _description_
        """
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
            train=False,
            download=True,
            transform=transform,
        )

        # if Configs.subtrain:
        #     sampling_pool = np.arange(len(dataset))
        #     np.random.shuffle(sampling_pool)
        #     num_sampling = int(Configs.subtrain_ratio * len(dataset))
        #     sublist = list(sampling_pool[:num_sampling])
        #     dataset = torch.utils.data.Subset(dataset, sublist)
        #     print("==>> sampled dataset: ", len(dataset))

        # if Configs.subval:
        #     sampling_pool = np.arange(len(dataset))
        #     np.random.shuffle(sampling_pool)
        #     num_sampling = int(Configs.subval_ratio * len(dataset))
        #     sublist = list(sampling_pool[:num_sampling])
        #     dataset = torch.utils.data.Subset(dataset, sublist)
        #     print("==>> sampled dataset: ", len(dataset))

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=Configs.workers,
            pin_memory=Configs.pin_memory,
        )

        return loader

    # @staticmethod
    # def get_imagenet_dataloader(batch_size, data_dir, data_category="train"):
    #     import torch
    #     from torchvision import transforms
    #     from torchvision.datasets import ImageFolder

    #     if data_category == "train":
    #         use_training = True
    #         use_shuffle = True
    #     else:
    #         use_training = False
    #         use_shuffle = False

    #     # Define the transformations to be applied to the images
    #     transform = transforms.Compose(
    #         [
    #             transforms.Resize((224, 224)),  # Resize the images to a fixed size
    #             transforms.ToTensor(),  # Convert images to tensors
    #             transforms.Normalize(
    #                 mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    #             ),  # Normalize the image tensors
    #         ]
    #     )

    #     # Create an instance of the ImageFolder dataset
    #     dataset = ImageFolder(
    #         root=os.path.join(data_dir, data_category), transform=transform
    #     )

    #     class_names = dataset.classes
    #     # print(f"==>> class_names: {class_names}")
    #     class_to_idx = dataset.class_to_idx
    #     # print(f"==>> class_to_idx: {class_to_idx}")

    #     loader = torch.utils.data.DataLoader(
    #         dataset,
    #         batch_size=batch_size,
    #         shuffle=use_shuffle,
    #         num_workers=Configs.workers,
    #         pin_memory=Configs.pin_memory,
    #     )

    #     return loader

    # @staticmethod
    # def get_all_cifar10_data(resize=False, data_category="train"):
    #     """_summary_

    #     Args:
    #         resize (bool, optional): _description_. Defaults to False.
    #         data_category (str, optional): _description_. Defaults to "train".

    #     Returns:
    #         _type_: _description_
    #     """
    #     import torch
    #     import torchvision
    #     import torchvision.transforms as transforms

    #     if resize:
    #         transform = transforms.Compose(
    #             [
    #                 transforms.Resize((256, 256)),
    #                 transforms.RandomHorizontalFlip(),
    #                 transforms.RandomCrop(224, padding=0),
    #                 transforms.ToTensor(),
    #                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    #             ]
    #         )
    #     else:
    #         transform = transforms.Compose(
    #             [
    #                 transforms.ToTensor(),
    #                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    #             ]
    #         )

    #     if data_category == "train":
    #         use_training = True
    #         use_shuffle = True
    #     else:
    #         use_training = False
    #         use_shuffle = False

    #     dataset = torchvision.datasets.CIFAR10(
    #         root="./data/",
    #         train=False,
    #         download=True,
    #         transform=transform,
    #     )

    #     loader = torch.utils.data.DataLoader(
    #         dataset, batch_size=Configs.batch_size, shuffle=False
    #     )

    #     images_all = []
    #     labels_all = []

    #     for images, labels in loader:
    #         images_all.extend(images)
    #         labels_all.extend(labels)

    #     images_torch = torch.stack(images_all)
    #     print(f"==>> images_torch.shape: {images_torch.shape}")
    #     labels_torch = torch.stack(labels_all)
    #     print(f"==>> labels_torch.shape: {labels_torch.shape}")

    #     # images, labels = next(iter(loader))

    #     return images_torch, labels_torch

    @staticmethod
    def get_umap_edge_dataloader(batch_size, data_dir="./data", data_category="train"):
        """_summary_

        Args:
            batch_size (_type_): _description_
            data_dir (str, optional): _description_. Defaults to "./data".
            data_category (str, optional): _description_. Defaults to "train".

        Returns:
            _type_: _description_
        """

        dataloader = Data.get_cifar10_dataloader(
            batch_size, data_dir=data_dir, data_category=data_category
        )

        if data_category == "train":
            model = torch.hub.load("facebookresearch/dino:main", "dino_vits16")
            out_embeddings, out_labels = GetDINOEmbedding.get_embedding_with_dinov2(
                dataloader, model, device="cuda"
            )
            out_embeddings = torch.from_numpy(out_embeddings)
            out_labels = torch.from_numpy(out_labels)
            print(f"==>> out_embeddings.shape: {out_embeddings.shape}")
            print(f"==>> out_labels.shape: {out_labels.shape}")

            images_torch = out_embeddings.clone().detach().cpu()
            labels_torch = out_labels.clone().detach().cpu()

            images_all = []
            labels_all = []
            for images, labels in dataloader:
                images_all.extend(images)
                labels_all.extend(labels)
            images_all_torch = torch.stack(images_all)
            print(f"==>> images_all_torch.shape: {images_all_torch.shape}")
            labels_all_torch = torch.stack(labels_all)
            print(f"==>> labels_all_torch.shape: {labels_all_torch.shape}")

            graph = DataUtils.get_umap_graph(
                images_torch.numpy(),
                n_neighbors=Configs.umap_n_neighbors,
                metric=Configs.umap_metric,
                random_state=Configs.umap_random_state,
            )
            umap_edge_dataset = UMAPDataset(images_all_torch, graph, n_epochs=200)

            # sampling_pool = np.arange(len(umap_edge_dataset))
            # np.random.shuffle(sampling_pool)
            # num_sampling = int(0.5 * len(umap_edge_dataset))
            # sublist = list(sampling_pool[:num_sampling])
            # umap_edge_dataset = torch.utils.data.Subset(umap_edge_dataset, sublist)
            # print("==>> sampled dataset: ", len(umap_edge_dataset))

            loader = torch.utils.data.DataLoader(
                umap_edge_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=Configs.workers,
                pin_memory=Configs.pin_memory,
            )

            return loader, images_torch, labels_torch
        else:
            loader = dataloader

        return loader, None, None
