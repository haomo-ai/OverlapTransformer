import os
import sys
p = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
if p not in sys.path:
    sys.path.append(p)
import matplotlib.pyplot as plt
import numpy as np
from com_overlap import com_overlap
import yaml
from tools.utils.utils import *
from modules.overlap_transformer import featureExtracter
import torch

# load config ================================================================
config_filename = '../config/config.yml'
config = yaml.safe_load(open(config_filename))
calib_file = config["demo1_config"]["calib_file"]
poses_file = config["demo1_config"]["poses_file"]
test_weights = config["demo1_config"]["test_weights"]
# ============================================================================

# load scan paths
scan_folder = "./scans"
scan_paths = load_files(scan_folder)

# load calibrations
T_cam_velo = load_calib(calib_file)
T_cam_velo = np.asarray(T_cam_velo).reshape((4, 4))
T_velo_cam = np.linalg.inv(T_cam_velo)

# load poses
poses = load_poses(poses_file)
pose0_inv = np.linalg.inv(poses[0])

# for KITTI dataset, we need to convert the provided poses
# from the camera coordinate system into the LiDAR coordinate system
poses_new = []
for pose in poses:
    poses_new.append(T_velo_cam.dot(pose0_inv).dot(pose).dot(T_cam_velo))
poses = np.array(poses_new)

# calculate overlap
ground_truth_mapping, current_range, reference_range_list = com_overlap(scan_paths, poses, frame_idx=0)

# build model and load pretrained weights
amodel = featureExtracter(channels=1, use_transformer=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
amodel.to(device)
print("Loading weights from ", test_weights)
checkpoint = torch.load(test_weights)
amodel.load_state_dict(checkpoint['state_dict'])
amodel.eval()

overlap_pos = round(ground_truth_mapping[1,-1]*100,2)
overlap_neg = round(ground_truth_mapping[2,-1]*100,2)

reference_range_pos = reference_range_list[1]
reference_range_neg = reference_range_list[2]
currentrange_neg_tensor = torch.from_numpy(current_range).unsqueeze(0)
currentrange_neg_tensor = currentrange_neg_tensor.unsqueeze(0).cuda()
reference_range_pos_tensor = torch.from_numpy(reference_range_pos).unsqueeze(0)
reference_range_pos_tensor = reference_range_pos_tensor.unsqueeze(0).cuda()
reference_range_neg_tensor = torch.from_numpy(reference_range_neg).unsqueeze(0)
reference_range_neg_tensor = reference_range_neg_tensor.unsqueeze(0).cuda()

# generate descriptors
des_cur = amodel(currentrange_neg_tensor).cpu().detach().numpy()
des_pos = amodel(reference_range_pos_tensor).cpu().detach().numpy()
des_neg = amodel(reference_range_neg_tensor).cpu().detach().numpy()

# calculate similarity
dis_pos = np.linalg.norm(des_cur - des_pos)
dis_neg = np.linalg.norm(des_cur - des_neg)
sim_pos = round(1/(1+dis_pos),2)
sim_neg = round(1/(1+dis_neg),2)

plt.figure(figsize=(8,4))
plt.subplot(311)
plt.title("query: " + scan_paths[0])
plt.imshow(current_range)
plt.subplot(312)
plt.title("positive reference: " + scan_paths[1] +  " - similarity: " + str(sim_pos))
plt.imshow(reference_range_list[1])
plt.subplot(313)
plt.title("negative reference: " + scan_paths[2] +  " - similarity: " + str(sim_neg))
plt.imshow(reference_range_list[2])
plt.show()
