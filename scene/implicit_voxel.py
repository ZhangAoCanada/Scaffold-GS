import numpy as np
from PIL import Image
import torch
from torch import nn
# NOTE: for depth anything
import cv2
from depth_anything_v2.dpt import DepthAnythingV2

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

class ImplicitVoxel(nn.Module):
    def __init__(self, 
                encoder_name='vitl', 

                H=900, W=1600,
                resize_lim=(0.193, 0.225),
                final_dim=(128, 352),
                bot_pct_lim=(0.0, 0.22),
                rot_lim=(-5.4, 5.4),
                rand_flip=True,

                xbound=[-50.0, 50.0, 0.5],
                ybound=[-50.0, 50.0, 0.5],
                zbound=[-50.0, 50.0, 0.5],
                dbound=[1.0, 50.0, 1.0],
                ):
        super(ImplicitVoxel, self).__init__()

        # NOTE: pretrained dinov2 (for encoder)
        # self.pretrained = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')

        # NOTE: depthanythingv2
        self.depth_model = DepthAnythingV2(**model_configs[encoder_name])
        self.depth_model.load_state_dict(torch.load(f'Depth-Anything-V2/checkpoints/depth_anything_v2_{encoder_name}.pth', map_location=DEVICE))
        self.depth_model = self.depth_model.to(DEVICE)

        # NOTE: LSS
        self.grid_conf = {
            'xbound': xbound,
            'ybound': ybound,
            'zbound': zbound,
            'dbound': dbound,
        }
        self.data_aug_conf = {
            'H': H,
            'W': W,
            'resize_lim': resize_lim,
            'final_dim': final_dim,
            'bot_pct_lim': bot_pct_lim,
            'rot_lim': rot_lim,
            'rand_flip': rand_flip,
        }
        dx, bx, nx = self.gen_dx_bx(xbound, ybound, zbound)
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)
        self.downsample = 16
        self.points = None


    def gen_dx_bx(self, xbound, ybound, zbound):
        dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
        bx = torch.Tensor([row[0] + row[2]/2.0 for row in [xbound, ybound, zbound]])
        nx = torch.LongTensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]])
        return dx, bx, nx

    def create_frustum(self, image_shape):
        # make grid in image plane
        ogfH, ogfW = image_shape[0], image_shape[1] # image_shape: (H, W)
        fH, fW = ogfH // self.downsample, ogfW // self.downsample
        ds = torch.arange(*self.grid_conf['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        D, _, _ = ds.shape
        xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
        ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)
        # D x H x W x 3
        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False)
    
    def get_geometry(self, R, T, K):
        points = self.frustum # D x H x W x 3
        # cam to ego
        points = torch.cat((points[..., :2] * points[..., 2:3], points[..., 2:3]), -1)
        combine = R.matmul(torch.inverse(K))
        points = combine.view(1, 1, 1, 3, 3).matmul(points.unsqueeze(-1)).squeeze(-1)
        points = points + T.view(1, 1, 1, 3)
        return points
    
    def depth_forward(self, image):
        """
        Convert depth to (D, H, W)
        """
        depth = self.depth_model.infer_image(image)
        return depth

    def forward(self, cams):
        """
        testing with depth anything
        """
        for cam in cams:
            image = np.array(cam.image)
            R = torch.tensor(cam.R)
            T = torch.tensor(cam.T)
            K = torch.tensor(cam.K)
            self.frustum = self.create_frustum(image.shape)
            depth_voxel = self.depth_forward(image)
            print(depth_voxel.shape) 
            print(self.frustum.shape)
        return depth_voxel