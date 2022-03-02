#!/usr/bin/env python3
# Developed by Junyi Ma, Xieyuanli Chen, and Jun Zhang
# This file is covered by the LICENSE file in the root of the project OverlapTransformer:
# https://github.com/haomo-ai/OverlapTransformer/
# Brief: Visualize evaluation on KITTI 00


import os
import sys
p = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
if p not in sys.path:
    sys.path.append(p)

from matplotlib import pyplot as plt
import torch
import os
import numpy as np
import sys
from modules.overlap_transformer import featureExtracter
from tools.read_samples import read_one_need_from_seq
np.set_printoptions(threshold=sys.maxsize)
from tools.utils.utils import *
import faiss
import yaml

"""
    Evaluation is conducted on KITTI 00 in our work.
    Args:
        amodel: pretrained model.
        data_root_folder: dataset root of KITTI.
        test_seq: "00" in our work.
"""
def test_chosen_seq(amodel, data_root_folder, test_seq, calib_file, poses_file, cov_file):
    T_cam_velo = load_calib(calib_file)
    T_cam_velo = np.asarray(T_cam_velo).reshape((4, 4))
    T_velo_cam = np.linalg.inv(T_cam_velo)
    poses = load_poses(poses_file)
    pose0_inv = np.linalg.inv(poses[0])

    poses_new = []
    for pose in poses:
        poses_new.append(T_velo_cam.dot(pose0_inv).dot(pose).dot(T_cam_velo))
    poses = np.array(poses_new)

    print(poses.shape)

    xyz = poses[:,0:4, -1]
    print(xyz.shape)

    covs = open(cov_file)
    covs = [overlap.replace('\n', '').split() for overlap in covs.readlines()]
    covs = np.asarray(covs, dtype=float)
    covs = covs.reshape((covs.shape[0], 6, 6))

    covs_x = np.sqrt(covs[:, 0, 0])
    covs_y = np.sqrt(covs[:, 1, 1])

    range_images = os.listdir(os.path.join(data_root_folder, test_seq, "depth_map"))

    des_list = np.zeros((len(range_images), 256))
    des_list_inv = np.zeros((len(range_images), 256))

    """Calculate the descriptors of scans"""
    print("Calculating the descriptors of scans ...")
    for i in range(0, len(range_images)):
        current_batch = read_one_need_from_seq(data_root_folder, str(i).zfill(6), test_seq)
        current_batch_inv_double = torch.cat((current_batch, current_batch), dim=-1)
        current_batch_inv = current_batch_inv_double[:,:,:,450:1350]
        current_batch = torch.cat((current_batch, current_batch), dim=0)
        amodel.eval()
        current_batch_des = amodel(current_batch)
        des_list[i, :] = current_batch_des[0, :].cpu().detach().numpy()
        des_list_inv[i, :] = current_batch_des[1, :].cpu().detach().numpy()

    des_list = des_list.astype('float32')

    for i in range(101, 4541-1):
        nlist = 1
        k = 1
        d = 256
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
        assert not index.is_trained
        index.train(des_list[:i-100,:])
        assert index.is_trained
        index.add(des_list[:i-100,:])
        plt.clf()

        plt.scatter(xyz[(i+1):, 0], xyz[(i+1):, 1], linewidths=0.05, c="pink")
        plt.scatter(xyz[:i, 0], xyz[:i, 1], linewidths=0.05, c="yellow",alpha=0.6)
        plt.scatter(xyz[i, 0], xyz[i, 1], linewidths=5, c="black", label="current pose")
        search_space = max(covs_x[i], covs_y[i]) * 2
        theta = np.arange(0,2*3.14159,0.01)
        xc = np.cos(theta)
        yc = np.sin(theta)
        xc = xc * search_space + xyz[i, 0]
        yc = yc * search_space + xyz[i, 1]
        plt.scatter(xc, yc, s=0.0001, c="blue", alpha=0.8)
        plt.scatter(100000, 100000, s=1, c="blue", alpha=0.8, label = "search space")
        plt.scatter(100000, 100000, linewidths=0.05, c="red", alpha=0.8, label="loop candidate")

        """Faiss searching"""
        D, I = index.search(des_list[i, :].reshape(1, -1), k)
        for j in range(D.shape[1]):
            """The nearest 100 frames are not considered."""
            if (i-I[:,j])<100:
                continue
            else:
                dis = np.sqrt(
                    (xyz[i, 0] - xyz[I[:, j], 0]) ** 2 + (xyz[i, 1] - xyz[I[:, j], 1]) ** 2)
                if dis < search_space:
                    plt.scatter(xyz[I[:, j], 0], xyz[I[:, j], 1], linewidths=0.05, c="red")

        plt.xlim([-50,600])
        plt.ylim([-325,325])
        plt.legend()
        plt.ion()
        plt.pause(0.01)
        plt.clf()


class testHandler():
    def __init__(self, height=64, width=900, channels=5, norm_layer=None, use_transformer=True,
                 data_root_folder=None,
                 test_seq=None, test_weights=None,
                 calib_file=None, poses_file=None, cov_file=None):
        super(testHandler, self).__init__()

        self.height = height
        self.width = width
        self.channels = channels
        self.norm_layer = norm_layer
        self.use_transformer = use_transformer
        self.data_root_folder = data_root_folder
        self.test_seq = test_seq
        self.calib_file = calib_file
        self.poses_file = poses_file
        self.cov_file = cov_file

        self.amodel = featureExtracter(channels=self.channels, use_transformer=self.use_transformer)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.amodel.to(self.device)

        self.parameters  = self.amodel.parameters()
        self.test_weights = test_weights
        self.overlap_thresh = 0.2

    def eval(self):
        with torch.no_grad():
            print("Loading weights from ", self.test_weights)
            checkpoint = torch.load(self.test_weights)
            self.amodel.load_state_dict(checkpoint['state_dict'])
            test_chosen_seq(self.amodel, self.data_root_folder, self.test_seq, self.calib_file, self.poses_file, self.cov_file)




if __name__ == '__main__':

    # load config ================================================================
    config_filename = '../config/config.yml'
    config = yaml.safe_load(open(config_filename))
    data_root_folder = config["data_root"]["data_root_folder"]
    test_seq = config["test_config"]["test_seqs"][0]
    test_weights = config["test_config"]["test_weights"]
    calib_file = config["viz_config"]["calib_file"]
    poses_file = config["viz_config"]["poses_file"]
    cov_file = config["viz_config"]["cov_file"]
    # ============================================================================

    """
        testHandler to handle with testing process.
        Args:
            height: the height of the range image (the beam number for convenience).
            width: the width of the range image (900, alone the lines of OverlapNet).
            channels: 1 for depth only in our work.
            norm_layer: None in our work for better model.
            use_transformer: Whether to use MHSA.
            data_root_folder: root of KITTI sequences. It's better to follow our file structure.
            test_seq: "00" in the evaluation.
            test_weights: pretrained weights.
    """
    test_handler = testHandler(height=64, width=900, channels=1, norm_layer=None, use_transformer=True,
                               data_root_folder=data_root_folder, test_seq=test_seq, test_weights=test_weights,
                               calib_file=calib_file,poses_file=poses_file, cov_file=cov_file)
    test_handler.eval()


