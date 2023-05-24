""" >>> +++ use timm to get pytorch models; """

""" +++ +++ get the model list with timm; """
import torch
import timm

listed_models = timm.list_models("*")
# print("==>> listed_models: ", listed_models)
print("==>> listed_models len: ", len(listed_models))


listed_pretraiend_models = timm.list_models(pretrained=True)
# print("==>> listed_pretraiend_models: ", listed_pretraiend_models)
print("==>> listed_pretraiend_models len: ", len(listed_pretraiend_models))

convnext_listed_models = timm.list_models("convnextv2*")
print("==>> convnext_listed_models: ", convnext_listed_models)
print("==>> convnext_listed_models len: ", len(convnext_listed_models))


""" +++ +++ load the needed model; """
used_model = timm.create_model("convnextv2_atto", pretrained=True)
# print("==>> used_model: ", used_model)
model_config = used_model.default_cfg
print("==>> model_config: ", model_config)

""" +++ +++ save the model pretrained weights from timm; """
# torch.save(
#     used_model.state_dict(), "/data/SSD1/data/weights/convnextv2_atto_timm_xinma.pt"
# )

""" +++ +++ load the pretrained weights to the timm model;"""
# weights_path = "/data/SSD1/data/weights/convnextv2_atto_1k_224_ema.pt"
# weights_dict = torch.load(weights_path)["model"]
# print("==>> weights_dict: ", weights_dict.keys())
# loading_ret = used_model.load_state_dict(weights_dict)
# print("==>> loading_ret: ", loading_ret)


""" +++ +++ download a image online; """
import urllib

url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
try:
    urllib.URLopener().retrieve(url, filename)
except:
    urllib.request.urlretrieve(url, filename)


""" +++ +++ test the pretrained model; """
from PIL import Image
from torchvision import transforms

model = used_model
input_image = Image.open(filename)
preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

if torch.cuda.is_available():
    input_batch = input_batch.to("cuda")
    model.to("cuda")

with torch.no_grad():
    output = model(input_batch)
# Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
# print(output[0])
# The output has unnormalized scores. To get probabilities, you can run a softmax on it.
probabilities = torch.nn.functional.softmax(output[0], dim=0)
# print(probabilities)


""" +++ download imagenet 1K labels to identify the prediciton; """
import os
import subprocess


# Example file path
file_path = "./imagenet_classes.txt"

# Check if the file exists
if os.path.exists(file_path):
    print("File exists")
else:
    print("File does not exist")
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    process = subprocess.Popen(["wget", url], stdout=subprocess.PIPE)
    output, error = process.communicate()
    print(output.decode())

with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())
