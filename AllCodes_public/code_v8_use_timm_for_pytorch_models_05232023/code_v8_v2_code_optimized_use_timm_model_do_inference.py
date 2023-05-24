import torch
import time


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
    def get_timm_model():

        import timm

        listed_models = timm.list_models("*")
        # print("==>> listed_models: ", listed_models)
        # print("==>> listed_models len: ", len(listed_models))

        listed_pretraiend_models = timm.list_models(pretrained=True)
        # print("==>> listed_pretraiend_models: ", listed_pretraiend_models)
        # print("==>> listed_pretraiend_models len: ", len(listed_pretraiend_models))

        convnext_listed_models = timm.list_models("convnextv2*")
        # print("==>> convnext_listed_models: ", convnext_listed_models)
        # print("==>> convnext_listed_models len: ", len(convnext_listed_models))

        """ +++ +++ load the needed model; """
        used_model = timm.create_model(
            "efficientformerv2_s2.snap_dist_in1k", pretrained=True
        )
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
            print(f"==>> output.shape: {output.shape}, time: {diff_time}")

        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top5_prob, top5_catid = torch.topk(probabilities, 5)
        for i in range(top5_prob.size(0)):
            print(classes[top5_catid[i]], top5_prob[i].item())


if __name__ == "__main__":

    ExprCommonSetting.generate_folders()

    model = TimmInference.get_timm_model()
    input_batch = TimmInference.get_input_data()
    classes = TimmInference.get_classes_info()

    TimmInference.inference_with_pretrained_model(model, input_batch, classes)
