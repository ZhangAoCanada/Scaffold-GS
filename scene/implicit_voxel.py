import numpy as np
from PIL import Image
import torch
from torch import nn
# NOTE: for depth anything
import cv2
from depthanything.depth_anything_v2.dpt import DepthAnythingV2

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

class ImplicitVoxel(nn.Module):
    def __init__(self, encoder_name='vitl', ):
        super(ImplicitVoxel, self).__init__()
        # # NOTE: pretrained dinov2 (for encoder)
        # self.pretrained = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
        # NOTE: depthanythingv2
        self.depth_model = DepthAnythingV2(**model_configs[encoder_name])
        self.depth_model.load_state_dict(torch.load('depthanything/depth_anything_v2/checkpoints/depthanything_v2_{encoder_name}.pth', map_location=DEVICE))
        self.depth_model = self.depth_model.to(DEVICE)

    
    def encoder(self,):
        return self.depth_model