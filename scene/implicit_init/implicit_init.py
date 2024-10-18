import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2

import torch
from torch import nn
from torchvision import models
from torchvision.models import list_models, resnet50, resnet101
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.transforms import v2 as transforms

from scene.implicit_init.tools import gen_dx_bx, cumsum_trick, QuickCumsum, QuickFind, cumsum_trick_all, QuickCumsumAll, QuickFindAll, quickfind_all
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, getWorld2View, fov2focal
from utils.general_utils import get_expon_lr_func
from scene.implicit_init.tools import ndc2pix, plot_points2d, plot_points, plot_voxel


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



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
    image = image.to(DEVICE)
    return image


def pil2tensor(cam):
    pil_image = cam.image
    w, h = cam.width, cam.height
    pil_image = pil_image.resize((w, h))
    image_np = np.array(pil_image)
    image_np = image_np / 255.
    image = torch.tensor(image_np, dtype=torch.float32, device=DEVICE).permute(2, 0, 1)
    return image


class ImplicitInit(nn.Module):
    def __init__(self, ):
        super(ImplicitInit, self).__init__()
        self.grid_conf = {
            'xbound': [-100., 100., 0.01],
            'ybound': [-100., 100., 0.01],
            'zbound': [-100., 100., 0.01],
        }
        # self.depth_dir = "depth_kitti"
        self.depth_dir = "depth"

        assert self.depth_dir in ["depth", "depth_kitti"]
        if self.depth_dir == "depth":
            self.max_depth = 20.
        elif self.depth_dir == "depth_kitti":
            self.max_depth = 80.
        else:
            raise ValueError(f"[ERROR] Invalid depth_dir: {self.depth_dir}")

        self.downsample_ratio = 16
        self.num_ratio = 0.5
        self.if_debug = False

        dx, bx, nx = gen_dx_bx(self.grid_conf['xbound'], self.grid_conf['ybound'], self.grid_conf['zbound'])
        self.dx = torch.tensor(dx, device=DEVICE)
        self.bx = torch.tensor(bx, device=DEVICE)
        self.nx = torch.tensor(nx, device=DEVICE)

        ### NOTE: for resnet ###
        self.return_nodes = {
            'layer1.2.bn3': 'layer1',
            'layer2.3.bn3': 'layer2',
            'layer3.5.bn3': 'layer3',
            'layer4.2.bn3': 'layer4',
        }
        self.resnet = resnet50(weights="IMAGENET1K_V2")
        self.resnet = self.resnet.to(DEVICE)
        self.resnet.eval()
        self.model = create_feature_extractor(self.resnet, self.return_nodes)

        self.features = None
        self.points = None
        self.colors = None

        self.points_repeatcount = None

        ### NOTE: use voxel ###
        # self.voxel = torch.zeros((self.nx[0], self.nx[1], self.nx[2], 3), device=DEVICE)
        # self.voxel_denom = torch.zeros((self.nx[0], self.nx[1], self.nx[2]), device=DEVICE)
    

    def get_features(self, image):
        image = image2tensor(image).unsqueeze(0)
        with torch.no_grad():
            features = self.model(image)
        return features
    

    def resized_layer2features(self, cam):
        image = cam.image
        # w, h = image.size
        w, h = cam.width, cam.height
        shape = (h, w)
        features = self.get_features(image)['layer1']
        features = torch.nn.functional.interpolate(features, size=shape, mode='bilinear', align_corners=False)
        return features


    def tensor_resize(self, tensor, downsample_ratio):
        """
        tensor: (W, H, C)
        downsample_ratio: int
        """
        tensor = tensor.permute(2, 1, 0)
        tensor = torch.nn.functional.interpolate(tensor.unsqueeze(0), scale_factor=1/downsample_ratio, mode='bilinear', align_corners=False).squeeze(0)
        tensor = tensor.permute(2, 1, 0)
        return tensor


    def depth2points(self, image, features, depth, intr_properties, cam=None):
        """
        Convert depth to (D, H, W)
        shape: frumstum shape (D, H, W)
        """
        C, H, W = features.shape
        ### NOTE: for debug
        if cam != None:
            image_np = image.permute(1, 2, 0).cpu().numpy() * 255.
            image_np = image_np.astype(np.uint8)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            depth_np = depth.detach().cpu().numpy() / self.max_depth * 255.
            depth_np = depth_np.astype(np.uint8)
            depth_merge = np.stack((depth_np, depth_np, depth_np), -1)
            debug_img = np.concatenate((image_np, depth_merge), axis=1)
            debug_dir = "./outputs/debug_depth"
            if not os.path.exists(debug_dir):
                os.makedirs(debug_dir)
            cv2.imwrite(f"{debug_dir}/{cam.image_name}.png", debug_img)

        image = image.permute(2, 1, 0)
        depth = depth.permute(1, 0)
        features = features.permute(2, 1, 0)
        cx = intr_properties['cx']
        cy = intr_properties['cy']
        fx = intr_properties['fx']
        fy = intr_properties['fy']

        x, y = torch.meshgrid(torch.arange(W, device=DEVICE), torch.arange(H, device=DEVICE))
        x = (x - cx) / fx
        y = (y - cy) / fy
        z = depth
        ################# NOTE: add downsample_ratio #################
        points = torch.stack((torch.mul(x, z), torch.mul(y, z), z), -1)
        points = self.tensor_resize(points, self.downsample_ratio)
        features = self.tensor_resize(features, self.downsample_ratio)
        image = self.tensor_resize(image, self.downsample_ratio)
        points = points.reshape(-1, 3)
        # points = torch.stack((torch.mul(x, z), torch.mul(y, z), z), -1).reshape(-1, 3)
        ##############################################################
        colors = image.reshape(-1, 3)
        features = features.reshape(-1, C)
        return colors, features, points

    
    def cam2world(self, points, R, T):
        W2C = torch.tensor(getWorld2View(R, T)).to(DEVICE)
        C2W = torch.inverse(W2C)

        # points = torch.cat([points, torch.ones(points.shape[0], 1, device=DEVICE)], dim=-1)
        # # points = C2W.matmul(points.t()).t()
        # points = C2W.matmul(points.unsqueeze(-1)).squeeze(-1)
        # points[:, :3] = points[:, :3] / points[:, 3:4]
        # return points[:, :3]

        R = C2W[:3, :3]
        T = C2W[:3, 3]
        points = R.matmul(points.t()).t() + T
        return points


    def points2voxel(self, colors, features, points):
        points = (points - self.bx) / self.dx
        points = points.round().long()

        # mask = (points >= 0).all(-1) & (points < (self.nx)).all(-1)
        mask = (points >= 0).all(-1) & (points < (self.nx - 2)).all(-1)
        points = points[mask]
        colors = colors[mask]

        ranks = points[:, 0] * (self.nx[1] * self.nx[2]) + points[:, 1] * self.nx[2] + points[:, 2]
        sorts = ranks.argsort()
        colors, features, points, ranks = colors[sorts], features[sorts], points[sorts], ranks[sorts]

        # colors, features, points = QuickFindAll.apply(colors, features, points, ranks)
        colors, features, points = quickfind_all(colors, features, points, ranks)

        points = points.float() * self.dx + self.bx
        return colors, features, points
    

    # def add2voxel(self, colors, features, points):
    #     points = (points - self.bx) / self.dx
    #     points = points.round().long()
    #     points_voxel = torch.zeros_like(self.voxel)
    #     points_voxel[points[:, 0], points[:, 1], points[:, 2]] = colors
    #     mask = points_voxel.sum(-1) * self.voxel.sum(-1) != 0
    #     # self.voxel[mask] = (self.voxel[mask] + points_voxel[mask]) / 2
    #     self.voxel[mask] = self.voxel[mask]
    #     self.voxel_denom[mask] += 1
    #     self.voxel[~mask] += points_voxel[~mask]


    def add2voxelpnts(self, colors, features, points):
        points = (points - self.bx) / self.dx
        points = points.round().long()

        self.points = (self.points - self.bx) / self.dx
        self.points = self.points.round().long()

        self.points = torch.cat((self.points, points), dim=0)
        self.colors = torch.cat((self.colors, colors), dim=0)
        self.features = torch.cat((self.features, features), dim=0)
        self.points_repeatcount = torch.cat((self.points_repeatcount, torch.zeros((points.shape[0], 1), device=DEVICE)), dim=0)

        ranks = self.points[:, 0] * (self.nx[1] * self.nx[2]) + self.points[:, 1] * self.nx[2] + self.points[:, 2]
        sorts = ranks.argsort()
        self.features, self.points, self.colors, ranks = self.features[sorts], self.points[sorts], self.colors[sorts], ranks[sorts]
        self.points_repeatcount = self.points_repeatcount[sorts]

        ranks_unique, inverse_indices, counts = torch.unique(ranks, sorted=True, return_inverse=True, return_counts=True)
        _, ind_sorted = torch.sort(inverse_indices, stable=True)
        cumsum = counts.cumsum(0)
        cumsum = torch.cat((torch.tensor([0], device=DEVICE), cumsum[:-1]))
        indices = ind_sorted[cumsum]
        assert ranks[indices].equal(ranks_unique)
        self.points = self.points[indices]
        self.colors = self.colors[indices]
        self.features = self.features[indices]
        self.points_repeatcount = self.points_repeatcount[indices]
        self.points_repeatcount += counts.unsqueeze(-1)

        # # self.colors, self.points = QuickFind.apply(self.colors, self.points, ranks)
        # self.colors, self.features, self.points = quickfind_all(self.colors, self.features, self.points, ranks)
        # print(f"self.points.shape: {self.points.shape}, points.shape: {points.shape}")

        self.points = self.points.float() * self.dx + self.bx


    def get_depth(self, cam, source_path):
        w, h = cam.width, cam.height
        depth_path = os.path.join(source_path, self.depth_dir, cam.image_name + ".png")
        depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        depth_image = cv2.resize(depth_image, (w, h), interpolation=cv2.INTER_NEAREST)
        depth_image = depth_image.astype(np.float32) / 255.0 * self.max_depth
        depth_image = torch.tensor(depth_image, device=DEVICE)
        return depth_image
    

    def get_focal(self, cam, shape=None): # shape: (H, W)
        fy = fov2focal(cam.FovY, cam.height)
        fx = fov2focal(cam.FovX, cam.width)
        if shape == None:
            return {
                'fx': fx,
                'fy': fy,
                'cx': cam.width / 2,
                'cy': cam.height / 2,
            }
        else:
            return {
                'fx': fx,
                'fy': fy,
                'cx': shape[1] / 2,
                'cy': shape[0] / 2,
            }


    def forward(self, cams, source_path):
        for cam in tqdm(cams):
            features = self.resized_layer2features(cam).squeeze(0)
            depth = self.get_depth(cam, source_path)
            image_tensor = pil2tensor(cam)
            intr_properties = self.get_focal(cam)
            colors, features, points = self.depth2points(image_tensor, features, depth, intr_properties, cam=cam)
            # plot_points(points, colors, "depth2points", cam.image_name)
            points = self.cam2world(points, cam.R, cam.T)
            # plot_points(points, colors, "cam2world", cam.image_name)
            colors, features, points = self.points2voxel(colors, features, points)
            # plot_points(points, colors, "points2voxel", cam.image_name)
            ######## NOTE: add to voxel points ########
            if self.features == None:
                self.features = torch.zeros((0, features.shape[1]), device=DEVICE)
                self.points = torch.zeros((0, 3), device=DEVICE)
                self.colors = torch.zeros((0, 3), device=DEVICE)
                self.points_repeatcount = torch.zeros((0, 1), device=DEVICE)
            self.add2voxelpnts(colors, features, points)
            ######## NOTE: add to voxel points #########
            # self.add2voxel(colors, features, points)
            ############################################
        #     print(f"[INFO] self.features: {self.features.shape}, self.points: {self.points.shape}, self.colors: {self.colors.shape}, self.points.max: {self.points.max(0), self.points.min(0)}")
        #     self.recover_points(points, colors, cam.R, cam.T, cam.FovX, cam.FovY, (cam.height, cam.width), cam=cam)
        #     self.recover_points(self.points, self.colors, cam.R, cam.T, cam.FovX, cam.FovY, (cam.height, cam.width), cam=cam, if_all=True)
        # self.recover_points(self.points, self.colors, cam.R, cam.T, cam.FovX, cam.FovY, (cam.height, cam.width), cam=None, if_all=True)
        # return self.colors, self.features, self.points


        self.points_repeatcount = self.points_repeatcount.squeeze(-1)
        repeatcount_sorted = self.points_repeatcount.clone().sort().values
        mask = self.points_repeatcount >= repeatcount_sorted[int(self.points_repeatcount.shape[0] * (1 - self.num_ratio))]
        self.colors = self.colors[mask]
        self.features = self.features[mask]
        self.points = self.points[mask]
        if self.if_debug:
            print("[INFO] Debugging to plot all points")
            for cam in tqdm(cams):
                self.recover_points(self.points, self.colors, cam.R, cam.T, cam.FovX, cam.FovY, (cam.height, cam.width), cam=cam)

        ########################################################
        # mask = self.voxel_denom != 0
        # self.voxel[mask] = self.voxel[mask]
        # points = torch.stack(torch.where(mask), -1) * self.dx + self.bx
        # colors = self.voxel[mask]
        # print("[INFO] Debugging to plot all points")
        # for cam in tqdm(cams):
        #     self.recover_points(points, colors, cam.R, cam.T, cam.FovX, cam.FovY, (cam.height, cam.width), cam=cam)
        ########################################################

        colors, features, points = self.colors.detach().cpu().numpy(), self.features.detach().cpu().numpy(), self.points.detach().cpu().numpy()
        np.save("outputs/colors.npy", colors)
        np.save("outputs/features.npy", features)
        np.save("outputs/points.npy", points)
        return colors, features, points


    def del_all(self):
        # delete everything and clear gpu memory
        del self.features
        del self.points
        del self.colors
        torch.cuda.empty_cache()
    

    def recover_points(self, points, colors, R, T, fov_x, fov_y, original_shape, cam=None, if_all=False):
        H, W = original_shape

        if if_all:
            pname = "recover_world_all"
        else:
            pname = "recover_world"
        if cam != None:
            plot_points(points, colors, pname, cam.image_name)
        else:
            plot_points(points, colors, pname)

        world_view_transform = torch.tensor(getWorld2View2(R, T)).transpose(0, 1).to(DEVICE)
        projection_matrix = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=fov_x, fovY=fov_y).transpose(0,1).to(DEVICE)
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0) # shaped [4, 4]

        points_cam = points.clone()
        w2c = world_view_transform.t()
        R = w2c[:3, :3]
        T = w2c[:3, 3]
        points_cam = R.matmul(points_cam.t()).t() + T

        points = torch.cat([points, torch.ones(points.shape[0], 1, device=DEVICE)], dim=-1)
        points = full_proj_transform.t().matmul(points.t()).t()
        points[:, :3] = points[:, :3] / points[:, 3:4]
        points = points[:, :3]

        if if_all:
            pname = "recover_viewpoint_all"
        else:
            pname = "recover_viewpoint"
        if cam != None:
            plot_points(points_cam, colors, pname, cam.image_name)
        else:
            plot_points(points_cam, colors, pname)

        # NOTE: to pixel
        S = torch.tensor([W, H], device=DEVICE)
        points = ndc2pix(points[:, :2], S)
        points = points.round().long()

        mask = (points[:, 0] >= 0) & (points[:, 0] < W) & (points[:, 1] >= 0) & (points[:, 1] < H)
        points = points[mask]
        colors = colors[mask]

        if if_all:
            pname = "recover_imagepnts_all"
        else:
            pname = "recover_imagepnts"
        if cam != None:
            plot_points(points_cam[mask], colors, pname, cam.image_name)
        else:
            plot_points(points_cam[mask], colors, pname)

        image = torch.zeros((H, W, 3), device=DEVICE)
        image[points[:, 1], points[:, 0]] = colors
        image = image.cpu().numpy()
        image = (image * 255).astype(np.uint8)
        image = Image.fromarray(image)

        if if_all:
            pname = "recover_image_all"
        else:
            pname = "recover_image"
        if cam != None:
            save_dir = os.path.join("outputs", pname)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            image.save(os.path.join(save_dir, f"{cam.image_name}.png"))
        else:
            image.save("outputs/recover_image.png")

        print("[DEBUG]")
