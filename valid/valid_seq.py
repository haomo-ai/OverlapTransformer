import os
import sys
p = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
if p not in sys.path:
    sys.path.append(p)
sys.path.append('../tools/')
sys.path.append('../modules/')    

import torch
import numpy as np
from tools.read_samples import read_one_need_from_seq
np.set_printoptions(threshold=sys.maxsize)
from tqdm import tqdm
from tools.utils.utils import *
import faiss
import time
import yaml



def validate_seq_faiss(amodel, seq_num):
    # load config ================================================================
    config_filename = '../config/config.yml'
    config = yaml.safe_load(open(config_filename))
    seqs_root = config["data_root"]["data_root_folder"]
    valid_scan_folder = config["data_root"]["valid_scan_folder"]
    ground_truth_folder = config["data_root"]["gt_valid_folder"]   # download needed *************************************************
    # calib_file_folder = config["data_root"]["calib_file_folder"]
    # pose_file_folder = config["data_root"]["pose_file_folder"]
    # ============================================================================
    scan_paths = load_files(valid_scan_folder)
    # calib_file = calib_file_folder + seq_num + "/calib.txt"
    # poses_file = pose_file_folder + seq_num + ".txt"
    # T_cam_velo = load_calib(calib_file)
    # T_cam_velo = np.asarray(T_cam_velo).reshape((4, 4))
    # T_velo_cam = np.linalg.inv(T_cam_velo)
    # poses = load_poses(poses_file)
    # pose0_inv = np.linalg.inv(poses[0])
    #
    # poses_new = []
    # for pose in poses:
    #     poses_new.append(T_velo_cam.dot(pose0_inv).dot(pose).dot(T_cam_velo))
    # poses = np.array(poses_new)

    loop_num = 0

    with torch.no_grad():
        des_list = np.zeros((len(scan_paths), 256))
        time11 = time.time()
        for i in range(len(scan_paths)):
            current_batch = read_one_need_from_seq(seqs_root, str(i).zfill(6), seq_num)
            current_batch = torch.cat((current_batch, current_batch), dim=0)
            amodel.eval()
            current_batch_des = amodel(current_batch)  # [1,256]
            des_list[i,:] = current_batch_des[0,:].cpu().detach().numpy()
        time22 = time.time()
        cal_time = (time22-time11)/len(scan_paths)

        des_list = des_list.astype('float32')
        used_num = 0
        print("calculated all descriptors")
        nlist = 1
        k = 3
        d = 256
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
        assert not index.is_trained
        index.train(des_list)
        assert index.is_trained
        index.add(des_list)

        for i in tqdm(range(0, len(scan_paths), 10)):
            used_num = used_num + 1
            gtm_path =ground_truth_folder + seq_num + "/overlap_"+str(i)+".npy"
            ground_truth_mapping =  np.load(gtm_path)
            time1 = time.time()
            D, I = index.search(des_list[i,:].reshape(1,-1), k)  # actual search
            time2 = time.time()
            time_diff = time2 - time1
            if I[:, 0] == i:
                # print("find itself")
                min_index = I[:, 1]
                min_value = D[:, 1]
            else:
                min_index = I[:, 0]
                min_value = D[:, 0]
            if ground_truth_mapping[min_index, 2] > 0.3:
                loop_num = loop_num + 1
    print("top1 rate: ", loop_num / used_num)
    return loop_num / used_num


