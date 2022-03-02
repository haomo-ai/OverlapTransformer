#!/usr/bin/env python3
# Developed by Junyi Ma, Xieyuanli Chen, and Jun Zhang
# This file is covered by the LICENSE file in the root of the project OverlapTransformer:
# https://github.com/haomo-ai/OverlapTransformer/
# Brief: read sampled range images of KITTI sequences as single input or batch input


import os
import sys
p = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
if p not in sys.path:
    sys.path.append(p)
    
import matplotlib.pyplot as plt
import torch
import cv2
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
from utils.utils import *
import yaml
from tools.read_all_sets import overlap_orientation_npz_file2string_string_nparray

"""
    read one needed $file_num range image from sequence $file_num.
    Args:
        data_root_folder: dataset root of KITTI.
        file_num: the index of the needed scan (zfill 6).
        seq_num: the sequence in which the needed scan is (zfill 2).
"""
def read_one_need_from_seq(data_root_folder, file_num, seq_num):

    depth_data = \
        np.array(cv2.imread(data_root_folder + seq_num + "/depth_map/" + file_num + ".png",
                            cv2.IMREAD_GRAYSCALE))

    depth_data_tensor = torch.from_numpy(depth_data).type(torch.FloatTensor).cuda()
    depth_data_tensor = torch.unsqueeze(depth_data_tensor, dim=0)
    depth_data_tensor = torch.unsqueeze(depth_data_tensor, dim=0)

    return depth_data_tensor


"""
    read one batch of positive samples and negative samples with respect to $f1_index in sequence $f1_seq.
    Args:
        data_root_folder: dataset root of KITTI.
        f1_index: the index of the needed scan (zfill 6).
        f1_seq: the sequence in which the needed scan is (zfill 2).
        train_imgf1, train_imgf2, train_dir1, train_dir2: the index dictionary and sequence dictionary following OverlapNet.
        train_overlap: overlaps dictionary following OverlapNet.
        overlap_thresh: 0.3 following OverlapNet.
"""
def read_one_batch_pos_neg(data_root_folder, f1_index, f1_seq, train_imgf1, train_imgf2, train_dir1, train_dir2, train_overlap, overlap_thresh):  # without end

    batch_size = 0
    for tt in range(len(train_imgf1)):
        if f1_index == train_imgf1[tt] and f1_seq == train_dir1[tt]:
            batch_size = batch_size + 1

    sample_batch = torch.from_numpy(np.zeros((batch_size, 1, 64, 900))).type(torch.FloatTensor).cuda()
    sample_truth = torch.from_numpy(np.zeros((batch_size, 1))).type(torch.FloatTensor).cuda()

    pos_idx = 0
    neg_idx = 0
    pos_num = 0
    neg_num = 0

    for j in range(len(train_imgf1)):
        pos_flag = False
        if f1_index == train_imgf1[j] and f1_seq==train_dir1[j]:
            if train_overlap[j]> overlap_thresh:
                pos_num = pos_num + 1
                pos_flag = True
            else:
                neg_num = neg_num + 1

            depth_data_r = \
                np.array(cv2.imread(data_root_folder + train_dir2[j] + "/depth_map/"+ train_imgf2[j] + ".png",
                            cv2.IMREAD_GRAYSCALE))

            depth_data_tensor_r = torch.from_numpy(depth_data_r).type(torch.FloatTensor).cuda()
            depth_data_tensor_r = torch.unsqueeze(depth_data_tensor_r, dim=0)

            if pos_flag:
                sample_batch[pos_idx,:,:,:] = depth_data_tensor_r
                sample_truth[pos_idx, :] = torch.from_numpy(np.array(train_overlap[j])).type(torch.FloatTensor).cuda()
                pos_idx = pos_idx + 1
            else:
                sample_batch[batch_size-neg_idx-1, :, :, :] = depth_data_tensor_r
                sample_truth[batch_size-neg_idx-1, :] = torch.from_numpy(np.array(train_overlap[j])).type(torch.FloatTensor).cuda()
                neg_idx = neg_idx + 1


    return sample_batch, sample_truth, pos_num, neg_num



if __name__ == '__main__':
    # load config ================================================================
    config_filename = '../config/config.yml'
    config = yaml.safe_load(open(config_filename))
    seqs_root = config["data_root"]["data_root_folder"]
    # ============================================================================


    seq = "08"
    cur_frame_idx = "000887"
    current_frame = read_one_need_from_seq(seqs_root, cur_frame_idx, seq)

    traindata_npzfiles = [os.path.join(seqs_root, seq, 'overlaps/train_set.npz')]
    (train_imgf1, train_imgf2, train_dir1, train_dir2, train_overlap) = \
        overlap_orientation_npz_file2string_string_nparray(traindata_npzfiles)
    reference_frames, reference_gts, pos_num, neg_num = read_one_batch_pos_neg(seqs_root, cur_frame_idx, seq,
                                              train_imgf1, train_imgf2, train_dir1, train_dir2, train_overlap, 0.3)


    # visualization
    print("the size of current_frame: ", current_frame.size())
    plt.figure(figsize=(15,3))
    plt.title("One sampled range image from KITTI sequence " + seq + ": " + cur_frame_idx + ".bin")
    plt.imshow(current_frame.cpu().detach().numpy()[0, 0, :, :])
    plt.show()

    print("the size of reference_frames: ", reference_frames.size())
    vis_idx = 5 # show the 5th sampled range image in the reference batch
    plt.figure(figsize=(15,3))
    plt.suptitle("One sampled query-reference from KITTI sequence " + seq + ", Overlap: " + str(reference_gts[vis_idx].item()))
    plt.subplot(211)
    plt.title("query")
    plt.imshow(current_frame.cpu().detach().numpy()[0, 0, :, :])
    plt.subplot(212)
    plt.title("reference")
    plt.imshow(reference_frames.cpu().detach().numpy()[vis_idx, 0, :, :])
    plt.show()


