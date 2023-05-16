# pip install ultralytics


from ultralytics import YOLO
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import cv2

# from google.colab.patches import cv2_imshow

model = YOLO("yolov8n.pt")

response = requests.get(
    "https://images.unsplash.com/photo-1600880292203-757bb62b4baf?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=2070&q=80"
)
image = Image.open(BytesIO(response.content))
image = np.asarray(image)


results = model.predict(image)
# print(f"==>> results: {results}")
print(results[0].boxes.cls)
