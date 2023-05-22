# pip install ultralytics


import numpy as np
from PIL import Image
from ultralytics import YOLO


class DataProcessing:
    @staticmethod
    def get_oneline_image():
        import urllib

        # url, file_path = (
        #     "https://github.com/pytorch/hub/raw/master/images/dog.jpg",
        #     "./work_dirs/dog.jpg",
        # )

        url, file_path = (
            "https://github.com/pytorch/hub/raw/master/images/deeplab1.png",
            "./work_dirs/deeplab1.png",
        )

        try:
            urllib.URLopener().retrieve(url, file_path)
        except:
            urllib.request.urlretrieve(url, file_path)

        return file_path


class UseYOLOv8:
    @staticmethod
    def inference_with_yolov8(image_path):

        image = Image.open(image_path).convert("RGB")
        image = np.asarray(image)

        model = YOLO("yolov8n.pt")

        results = model.predict(image)

        class_names_dict = results[0].names

        cls_prediction = results[0].boxes.cls
        bbox_prediction = results[0].boxes.xyxy
        conf_prediction = results[0].boxes.conf

        return cls_prediction, bbox_prediction, conf_prediction, class_names_dict


class ExprCommonSetting:
    @staticmethod
    def generate_folders():
        import os

        data_folder = "./data/"
        work_dirs_folder = "./work_dirs/"

        os.makedirs(data_folder, exist_ok=True)
        os.makedirs(work_dirs_folder, exist_ok=True)


if __name__ == "__main__":
    ExprCommonSetting.generate_folders()

    file_path = DataProcessing.get_oneline_image()

    (
        cls_prediction,
        bbox_prediction,
        conf_prediction,
        class_names_dict,
    ) = UseYOLOv8.inference_with_yolov8(file_path)

    for class_idx in cls_prediction.cpu().numpy():
        cls_prediction = class_names_dict[class_idx]
        print(f"==>> cls_prediction: {cls_prediction}")
    print(f"==>> bbox_prediction: {bbox_prediction}")
    print(f"==>> conf_prediction: {conf_prediction}")
