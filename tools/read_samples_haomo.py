#!/usr/bin/env python3
# Developed by Junyi Ma, Xieyuanli Chen, and Jun Zhang
# This file is covered by the LICENSE file in the root of the project OverlapTransformer:
# https://github.com/haomo-ai/OverlapTransformer/
# Brief: read sampled range images of Haomo dataset as single input or batch input


import os
import sys
p = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
if p not in sys.path:
    sys.path.append(p)
sys.path.append('../tools/')
sys.path.append('../modules/')
    
import torch
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
from utils.utils import *
import yaml
import matplotlib.pyplot as plt


def read_one_need_from_seq(data_root_folder, file_num):

    depth_data = np.load(data_root_folder + file_num + ".npy")
    depth_data_tensor = torch.from_numpy(depth_data).type(torch.FloatTensor).cuda()
    depth_data_tensor = torch.unsqueeze(depth_data_tensor, dim=0)
    depth_data_tensor = torch.unsqueeze(depth_data_tensor, dim=0)

    return depth_data_tensor

def read_one_need_from_seq_test(data_root_folder_test, file_num):

    depth_data = np.load(data_root_folder_test + file_num + ".npy")

    depth_data_tensor = torch.from_numpy(depth_data).type(torch.FloatTensor).cuda()
    depth_data_tensor = torch.unsqueeze(depth_data_tensor, dim=0)
    depth_data_tensor = torch.unsqueeze(depth_data_tensor, dim=0)

    return depth_data_tensor




def read_one_batch_pos_neg(data_root_folder, f1_index, f1_seq, train_imgf1, train_imgf2, train_dir1, train_dir2, train_overlap, overlap_thresh):  # without end

    batch_size = 0
    for tt in range(len(train_imgf1)):
        if f1_index == train_imgf1[tt] and f1_seq == train_dir1[tt] and (train_overlap[tt]> overlap_thresh or train_overlap[tt]<(overlap_thresh-0.0)): # TODO: You can update the range
            batch_size = batch_size + 1

    sample_batch = torch.from_numpy(np.zeros((batch_size, 1, 32, 900))).type(torch.FloatTensor).cuda()
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
            elif train_overlap[j]< overlap_thresh:
                neg_num = neg_num + 1
            else:
                continue

            depth_data_r = np.load(data_root_folder + train_imgf2[j] + ".npy")
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
    config_filename = '../config/config_haomo.yml'
    config = yaml.safe_load(open(config_filename))
    data_root_folder = config["file_root"]["data_root_folder"]
    triplets_for_training = config["file_root"]["triplets_for_training"]
    training_seqs = config["training_config"]["training_seqs"]
    # ============================================================================

    train_set_imgf1_imgf2_overlap = np.load(triplets_for_training)
    # print(train_set_imgf1_imgf2_overlap)

    cur_frame_idx = "003430"
    current_frame = read_one_need_from_seq(data_root_folder, cur_frame_idx)

    train_imgf1 = train_set_imgf1_imgf2_overlap[:, 0]
    train_imgf2 = train_set_imgf1_imgf2_overlap[:, 1]
    train_dir1 = np.zeros((len(train_imgf1),))  # to use the same form as KITTI
    train_dir2 = np.zeros((len(train_imgf2),))
    train_overlap = train_set_imgf1_imgf2_overlap[:, 2].astype(float)
    reference_frames, reference_gts, pos_num, neg_num = read_one_batch_pos_neg \
        (data_root_folder, cur_frame_idx, 0, train_imgf1, train_imgf2, train_dir1,
         train_dir2, train_overlap, 0.3)



    # visualization
    print("the size of current_frame: ", current_frame.size())
    plt.figure(figsize=(15,3))
    plt.title("One sampled range image from Haomo dataset: " + cur_frame_idx + ".bin")
    plt.imshow(current_frame.cpu().detach().numpy()[0, 0, :, :])
    plt.show()

    print("the size of reference_frames: ", reference_frames.size())
    vis_idx = 5 # show the 2rd sampled range image in the reference batch
    plt.figure(figsize=(15,3))
    plt.suptitle("One sampled query-reference from Haomo dataset, Overlap: " + str(reference_gts[vis_idx].item()))
    plt.subplot(211)
    plt.title("query")
    plt.imshow(current_frame.cpu().detach().numpy()[0, 0, :, :])
    plt.subplot(212)
    plt.title("reference")
    plt.imshow(reference_frames.cpu().detach().numpy()[vis_idx, 0, :, :])
    plt.show()


