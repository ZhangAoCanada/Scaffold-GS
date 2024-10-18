import torch, torchvision
from torch import nn

from torchvision import models
from torchvision.models import list_models, resnet50, resnet101
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor

from torchvision.transforms import v2 as transforms

from PIL import Image
import numpy as np

import cv2


def image2tensor(pil_image, input_size=224):
    w, h = pil_image.size
    # h, w should be the multiple of input_size
    h_size = h // input_size * input_size
    w_size = w // input_size * input_size
    pil_image = pil_image.resize((w, h))
    image_np = np.array(pil_image)
    image_np = image_np / 255.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Resize((h_size, w_size)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image_np)
    return image, (h, w)


# def image2tensor(pil_image, input_size=224):
#     h, w = pil_image.size
#     # h, w should be the multiple of input_size
#     h_size = h // input_size * input_size
#     w_size = w // input_size * input_size
#     pil_image = pil_image.resize((w, h))
#     image_np = np.array(pil_image)
#     image_np = image_np / 255.
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.ConvertImageDtype(torch.float32),
#         transforms.Resize((w_size, h_size)),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])
#     image = transform(image_np)
#     return image, (h, w)


class ResNet(nn.Module):
    def __init__(self, weights="IMAGENET1K_V2"):
        super(ResNet, self).__init__()
        self.model = resnet50(weights=weights)
        self.model.eval()
        self.return_nodes = {
                'layer1.2.bn3': 'layer1',
                'layer2.3.bn3': 'layer2',
                'layer3.5.bn3': 'layer3',
                'layer4.2.bn3': 'layer4',
            }
        # self.return_nodes = {
        #         'layer1': 'layer1',
        #         'layer2': 'layer2',
        #         'layer3': 'layer3',
        #         'layer4': 'layer4',
        #     }
        self.body = create_feature_extractor(self.model, self.return_nodes)
        
    def forward(self, x):
        with torch.no_grad():
            x = self.body(x)
        return x



# List available models
all_models = list_models()
classification_models = list_models(module=torchvision.models)
print(all_models)

# get graph node names
# resnet_50 = resnet50(weights="IMAGENET1K_V2") # layer1, layer2, layer3, layer4
train_nodes, eval_nodes = get_graph_node_names(resnet50())


sample_image = "data/tandt_db/tandt/truck/images/000001.jpg"
image = Image.open(sample_image)
image, (h, w) = image2tensor(image)

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
img_cv2 = image.permute(1, 2, 0).cpu().numpy() * std + mean
img_cv2 = img_cv2 * 255.
img_cv2 = img_cv2.astype(np.uint8)
cv2.imwrite("outputs/debug_img.png", img_cv2)

print(image.shape)
image = image.cuda().unsqueeze(0)
resnet = ResNet().cuda()
output = resnet(image)

for k, v in output.items():
    print(k, v.shape)
