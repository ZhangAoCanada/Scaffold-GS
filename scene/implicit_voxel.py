import numpy as np
from PIL import Image
import torch
from torch import nn

class ImplicitVoxel(nn.Module):
    def __init__(self, ):
        super(ImplicitVoxel, self).__init__()
        self.pretrained = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
    
    def encoder(self,):
        return self.pretrained