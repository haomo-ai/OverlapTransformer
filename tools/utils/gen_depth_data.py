#!/usr/bin/env python3
# Developed by Xieyuanli Chen and Thomas Läbe
# This file is covered by the LICENSE file in the root of the project OverlapNet:
#https://github.com/PRBonn/OverlapNet
# Brief: a script to generate depth data
import os
# from .utils import load_files
# import numpy as np
# from .utils import range_projection

from utils import load_files
import numpy as np
from utils import range_projection

import cv2
try:
    from utils import *
except:
    from utils import *

import scipy.linalg as linalg
def rotate_mat( axis, radian):
    rot_matrix = linalg.expm(np.cross(np.eye(3), axis / linalg.norm(axis) * radian))
    return rot_matrix
    # print(type(rot_matrix))



def gen_depth_data(scan_folder, dst_folder, normalize=False):
    """ Generate projected range data in the shape of (64, 900, 1).
      The input raw data are in the shape of (Num_points, 3).
  """
    # specify the goal folder
    dst_folder = os.path.join(dst_folder, 'depth')
    try:
        os.stat(dst_folder)
        print('generating depth data in: ', dst_folder)
    except:
        print('creating new depth folder: ', dst_folder)
        os.mkdir(dst_folder)

    # load LiDAR scan files
    scan_paths = load_files(scan_folder)

    depths = []
    axis_x, axis_y, axis_z = [1,0,0], [0,1,0], [0, 0, 1]#分别是x,y和z轴,也可以自定义旋转轴

    # iterate over all scan files
    for idx in range(len(scan_paths)):
        # load a point cloud
        current_vertex = np.fromfile(scan_paths[idx], dtype=np.float32)
        current_vertex = current_vertex.reshape((-1, 4))  # 四维分别是 x y z intensity

        current_vertex_xyz = current_vertex[:,0:3]
        current_vertex_xyz = current_vertex_xyz.T


        #  rotation
        reso = 1
        len_rot = int(360 / reso)
        for j in range(0, len_rot):
            yaw = np.pi * reso / 180
            rot_matrix = rotate_mat(axis_z, yaw)  # 绕Z轴旋转
            current_vertex_xyz_rot = np.matmul(rot_matrix, current_vertex_xyz)  # 3* （-1）
            current_vertex[:,0:3] = current_vertex_xyz_rot.T

            proj_range, _, _, _ = range_projection(current_vertex)  # proj_ranges   from larger to smaller

            # normalize the image
            if normalize:
                proj_range = proj_range / np.max(proj_range)

            # generate the destination path
            dst_path = os.path.join(dst_folder, str(idx).zfill(6))

            # save the semantic image as format of .npy
            # np.save(dst_path, proj_range)
            filename = "/home/mjy/dev/OverlapNet++/Rotation_Invariant_Valid/rotated_scans/"+\
                       scan_paths[idx][-10:-4] + "_" + str(j)+".png"
            cv2.imwrite(filename, proj_range)
            print(filename)
            # print('finished generating depth data at: ', dst_path)

    return depths


if __name__ == '__main__':
    scan_folder = '/home/mjy/dev/OverlapNet++/Rotation_Invariant_Valid/scans'
    dst_folder = '/home/mjy/dev/OverlapNet++/Rotation_Invariant_Valid/rotated_scans'

    depth_data = gen_depth_data(scan_folder, dst_folder)
