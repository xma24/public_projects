import torch
import time
import timm
import numpy as np


class ExprCommonSetting:
    @staticmethod
    def generate_folders():
        import os

        data_folder = "./data/"
        work_dirs_folder = "./work_dirs/"

        os.makedirs(data_folder, exist_ok=True)
        os.makedirs(work_dirs_folder, exist_ok=True)


class TimmInference:
    @staticmethod
    def search_timm_models(keywords="resnet*", pretrained=True):
        # listed_models = timm.list_models("*")
        # print("==>> listed_models: ", listed_models)
        # print("==>> listed_models len: ", len(listed_models))

        # listed_pretraiend_models = timm.list_models(pretrained=True)
        # print("==>> listed_pretraiend_models: ", listed_pretraiend_models)
        # print("==>> listed_pretraiend_models len: ", len(listed_pretraiend_models))

        listed_models = timm.list_models(keywords, pretrained=pretrained)
        print(f"==>> listed_models: {listed_models}")
        # print("==>> convnext_listed_models: ", convnext_listed_models)
        # print("==>> convnext_listed_models len: ", len(convnext_listed_models))

    @staticmethod
    def get_timm_model(model_name):

        """+++ +++ load the needed model;"""
        used_model = timm.create_model(model_name, pretrained=True)
        # print("==>> used_model: ", used_model)
        model_config = used_model.default_cfg
        # print("==>> model_config: ", model_config)

        return used_model

    @staticmethod
    def get_input_data():
        import urllib

        url, filename = (
            "https://github.com/pytorch/hub/raw/master/images/dog.jpg",
            "./work_dirs/dog.jpg",
        )
        try:
            urllib.URLopener().retrieve(url, filename)
        except:
            urllib.request.urlretrieve(url, filename)

        """ +++ +++ test the pretrained model; """
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

        return input_batch

    @staticmethod
    def get_classes_info():

        import urllib

        url, filename = (
            "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt",
            "./work_dirs/imagenet_classes.txt",
        )
        try:
            urllib.URLopener().retrieve(url, filename)
        except:
            urllib.request.urlretrieve(url, filename)

        with open(filename, "r") as f:
            classes = [s.strip() for s in f.readlines()]

        return classes

    @staticmethod
    def inference_with_pretrained_model(model, input_batch, classes):

        model.eval()

        if torch.cuda.is_available():
            input_batch = input_batch.to("cuda")
            model.to("cuda")

        with torch.no_grad():
            pre_time = time.time()
            output = model(input_batch)
            diff_time = time.time() - pre_time
            # print(f"==>> output.shape: {output.shape}, time: {diff_time}")

        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top5_prob, top5_catid = torch.topk(probabilities, 5)
        # for i in range(top5_prob.size(0)):
        #     print(classes[top5_catid[i]], top5_prob[i].item())

        return diff_time

    @staticmethod
    def calculate_infer_time(model_name, num_rounds):

        model_time_list = []
        for i in range(num_rounds):
            # model_name_1 = "efficientformerv2_s0.snap_dist_in1k"
            model = TimmInference.get_timm_model(model_name)
            model_infer_time = TimmInference.inference_with_pretrained_model(
                model, input_batch, classes
            )
            model_time_list.append(model_infer_time)
        model_time_array = np.array(model_time_list)
        model_time_mean = np.mean(model_time_array)
        return model_time_mean


if __name__ == "__main__":

    ExprCommonSetting.generate_folders()

    TimmInference.search_timm_models(keywords="*efficientnet*")
    TimmInference.search_timm_models(keywords="*efficientformer*")

    # import sys

    # sys.exit()

    input_batch = TimmInference.get_input_data()
    classes = TimmInference.get_classes_info()

    num_rounds = 10
    """ +++ test efficientformer; """
    # efficientformer_time_list = []
    # for i in range(num_rounds):
    #     model_name_1 = "efficientformerv2_s0.snap_dist_in1k"
    #     model_1 = TimmInference.get_timm_model(model_name_1)
    #     efficientformer_infer_time = TimmInference.inference_with_pretrained_model(
    #         model_1, input_batch, classes
    #     )
    #     efficientformer_time_list.append(efficientformer_infer_time)
    # efficientformer_time_array = np.array(efficientformer_time_list)
    # efficientformer_time_mean = np.mean(efficientformer_time_array)
    # model_name_1 = "efficientformerv2_s0.snap_dist_in1k"
    model_name_1 = "efficientformer_l1.snap_dist_in1k"
    efficientformer_time = TimmInference.calculate_infer_time(model_name_1, num_rounds)
    print(f"==>> efficientformer_time: {efficientformer_time / num_rounds}")

    """ +++ test efficientnet;"""
    model_name_2 = "efficientnet_b0.ra_in1k"
    efficientnet_time = TimmInference.calculate_infer_time(model_name_2, num_rounds)
    print(f"==>> efficientnet_time: {efficientnet_time / num_rounds}")

    """ +++ we know that efficientnet is faster and accurate than efficientformer."""
