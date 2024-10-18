import os
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
from functools import reduce
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

##################
import cv2


def pad_tensor(tensor_img, patch_size=14):
    """Pad tensor to be divisible by patch size
    """
    C, H, W = tensor_img.shape
    H_new = np.ceil(H / patch_size).astype(int) * patch_size
    W_new = np.ceil(W / patch_size).astype(int) * patch_size
    pad_H = H_new - H
    pad_W = W_new - W
    tensor_img = F.pad(tensor_img, (0, pad_W, 0, pad_H))
    return tensor_img.unsqueeze(0)


def to_pad_tensor(pil_image, patch_size=14):
    """Convert PIL image to tensor
    """
    tensor_img = torchvision.transforms.ToTensor()(pil_image)
    C, H, W = tensor_img.shape
    H_new = np.ceil(H / patch_size).astype(int) * patch_size
    W_new = np.ceil(W / patch_size).astype(int) * patch_size
    pad_H = H_new - H
    pad_W = W_new - W
    tensor_img = F.pad(tensor_img, (0, pad_W, 0, pad_H))
    return tensor_img.unsqueeze(0)


def to_tensor(pil_image, patch_size=14):
    """Convert PIL image to tensor
    """
    w, h = pil_image.size
    w_new = w // patch_size * patch_size
    h_new = h // patch_size * patch_size
    pil_image = pil_image.resize((w_new, h_new))
    return torchvision.transforms.ToTensor()(pil_image).unsqueeze(0) 


def unpad_tensor(tensor_img, H, W):
    """Unpad tensor
    """
    return tensor_img[:, :, :H, :W]


def ego_to_cam(points, rot, trans, intrins):
    """Transform points (3 x N) from ego frame into a pinhole camera
    """
    points = points - trans.unsqueeze(1)
    points = rot.permute(1, 0).matmul(points)

    points = intrins.matmul(points)
    points[:2] /= points[2:3]

    return points


def cam_to_ego(points, rot, trans, intrins):
    """Transform points (3 x N) from pinhole camera with depth
    to the ego frame
    """
    points = torch.cat((points[:2] * points[2:3], points[2:3]))
    points = intrins.inverse().matmul(points)

    points = rot.matmul(points)
    points += trans.unsqueeze(1)

    return points


def ndc2pix(v, S):
    return ((v + 1) * S - 1) / 2


def plot_points2d(points, colors, name="points"):
    fig = plt.figure()
    ax = fig.add_subplot()
    points_np = points.cpu().numpy()
    colors_np = colors.cpu().numpy()
    ax.scatter(points_np[:, 0], points_np[:, 1], s=0.2, c=colors_np)
    plt.savefig(f"outputs/{name}.png")


def plot_points(points, colors, name="points", image_name=None):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    points_np = points.detach().cpu().numpy()
    colors_np = colors.detach().cpu().numpy()
    # np.save("outputs/points.npy", points_np)
    # np.save("outputs/colors.npy", colors_np)
    # ax.scatter(points_np[:, 0], points_np[:, 1], points_np[:, 2], s=0.2, c=colors_np)
    ax.scatter(points_np[:, 0], points_np[:, 2], -points_np[:, 1], s=0.2, c=colors_np)
    if image_name == None:
        plt.savefig(f"outputs/{name}.png")
    else:
        save_dir = f"outputs/{name}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(f"{save_dir}/{image_name}.png")


def plot_voxel(voxel, name="selfvoxel"):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    voxel = voxel.cpu().numpy()
    nx, ny, nz, _ = voxel.shape
    x, y, z = np.indices((nx+1, ny+1, nz+1))
    voxel_mask = voxel.mean(-1)
    voxel_mask = np.where(voxel_mask > 0, 0.5, 0.)
    ax.voxels(x, z, -y, voxel_mask, facecolors=voxel)
    ax.set(xlabel='x', ylabel='y', zlabel='z')
    plt.savefig(f"outputs/{name}.png")



def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2]/2.0 for row in [xbound, ybound, zbound]])
    nx = torch.LongTensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]])
    return dx, bx, nx


def cumsum_trick(x, geom_feats, ranks):
    x = x.cumsum(0)
    kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
    kept[:-1] = (ranks[1:] != ranks[:-1])

    x, geom_feats = x[kept], geom_feats[kept]
    x = torch.cat((x[:1], x[1:] - x[:-1]))

    return x, geom_feats


class QuickCumsum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, geom_feats, ranks):
        x = x.cumsum(0)
        kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        kept[:-1] = (ranks[1:] != ranks[:-1])

        x, geom_feats = x[kept], geom_feats[kept]
        x = torch.cat((x[:1], x[1:] - x[:-1]))

        # save kept for backward
        ctx.save_for_backward(kept)

        # no gradient for geom_feats
        ctx.mark_non_differentiable(geom_feats)

        return x, geom_feats

    @staticmethod
    def backward(ctx, gradx, gradgeom):
        kept, = ctx.saved_tensors
        back = torch.cumsum(kept, 0)
        back[kept] -= 1

        val = gradx[back]

        return val, None, None


class QuickFind(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, geom_feats, ranks):
        # x = x.cumsum(0)
        kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        kept[:-1] = (ranks[1:] != ranks[:-1])

        x, geom_feats = x[kept], geom_feats[kept]
        # x = torch.cat((x[:1], x[1:] - x[:-1]))

        # save kept for backward
        ctx.save_for_backward(kept)

        # no gradient for geom_feats
        ctx.mark_non_differentiable(geom_feats)

        return x, geom_feats

    @staticmethod
    def backward(ctx, gradx, gradgeom):
        kept, = ctx.saved_tensors
        back = torch.cumsum(kept, 0)
        back[kept] -= 1

        val = gradx[back]

        return val, None, None


def cumsum_trick_all(colors, features, points, ranks):
    features = features.cumsum(0)
    kept = torch.ones(features.shape[0], device=features.device, dtype=torch.bool)
    kept[:-1] = (ranks[1:] != ranks[:-1])

    colors, features, points = colors[kept], features[kept], points[kept]
    features = torch.cat((features[:1], features[1:] - features[:-1]))

    return colors, features, points



class QuickCumsumAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, colors, features, points, ranks):
        features = features.cumsum(0)
        kept = torch.ones(features.shape[0], device=features.device, dtype=torch.bool)
        kept[:-1] = (ranks[1:] != ranks[:-1])

        colors, features, points = colors[kept], features[kept], points[kept]
        features = torch.cat((features[:1], features[1:] - features[:-1]))

        # save kept for backward
        ctx.save_for_backward(kept)

        # # no gradient for points
        # ctx.mark_non_differentiable(points)

        return colors, features, points

    @staticmethod
    def backward(ctx, gradx, gradgeom):
        kept, = ctx.saved_tensors
        back = torch.cumsum(kept, 0)
        back[kept] -= 1

        val = gradx[back]

        return val, None, None



def quickfind_all(colors, features, points, ranks):
    kept = torch.ones(features.shape[0], device=features.device, dtype=torch.bool)
    kept[:-1] = (ranks[1:] != ranks[:-1])

    colors, features, points = colors[kept], features[kept], points[kept]
    features = torch.cat((features[:1], features[1:] - features[:-1]))

    return colors, features, points



class QuickFindAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, colors, features, points, ranks):
        kept = torch.ones(features.shape[0], device=features.device, dtype=torch.bool)
        kept[:-1] = (ranks[1:] != ranks[:-1])

        colors, features, points = colors[kept], features[kept], points[kept]

        # # save kept for backward
        # ctx.save_for_backward(kept)

        # no gradient for geom_feats
        ctx.mark_non_differentiable(points)

        return colors, features, points

    @staticmethod
    def backward(ctx, gradx, gradgeom):
        kept, = ctx.saved_tensors
        back = torch.cumsum(kept, 0)
        back[kept] -= 1

        val = gradx[back]

        return val, None, None
