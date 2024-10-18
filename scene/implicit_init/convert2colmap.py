import numpy as np
import pycolmap

points = np.load("outputs/points.npy")
colors = np.load("outputs/colors.npy")

# # convert to colmap format
# colmap_points = []
# for i in range(points.shape[0]):
#     colmap_points.append(pycolmap.Point3D(points[i], colors[i]))

# # save to colmap .bin file
# pycolmap.write_points3D("outputs/points3D.bin", colmap_points)




# conver to ply file
from plyfile import PlyData, PlyElement

colors = colors * 255
colors = colors.astype(np.uint8)

vertex = np.zeros(points.shape[0], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
vertex['x'] = points[:, 0]
vertex['y'] = points[:, 1]
vertex['z'] = points[:, 2]
vertex['red'] = colors[:, 0]
vertex['green'] = colors[:, 1]
vertex['blue'] = colors[:, 2]

ply = PlyData([PlyElement.describe(vertex, 'vertex')], text=True)
ply.write("outputs/points.ply")


