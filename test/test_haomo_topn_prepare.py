#!/usr/bin/env python3
# Developed by Junyi Ma, Xieyuanli Chen, and Jun Zhang
# This file is covered by the LICENSE file in the root of the project OverlapTransformer:
# https://github.com/haomo-ai/OverlapTransformer/
# Brief: generate the prediction files for the following Recall@N calculation


import os
import sys
p = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
if p not in sys.path:
    sys.path.append(p)
sys.path.append('../tools/')
sys.path.append('../modules/')
    
import matplotlib.pyplot as plt
import torch
import yaml
import numpy as np

from modules.overlap_transformer_haomo import featureExtracter
from tools.read_samples_haomo import read_one_need_from_seq
from tools.read_samples_haomo import read_one_need_from_seq_test
np.set_printoptions(threshold=sys.maxsize)
from tqdm import tqdm
import faiss
from tools.utils.utils import *

class testHandler():
    def __init__(self, height=32, width=900, channels=1, norm_layer=None,use_transformer=True,
                 data_root_folder=None, data_root_folder_test=None, test_weights=None):
        super(testHandler, self).__init__()

        self.height = height
        self.width = width
        self.channels = channels
        self.norm_layer = norm_layer
        self.use_transformer = use_transformer
        self.data_root_folder = data_root_folder
        self.data_root_folder_test = data_root_folder_test

        self.amodel = featureExtracter(channels=self.channels, use_transformer=self.use_transformer)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.amodel.to(self.device)
        print(self.amodel)
        self.parameters  = self.amodel.parameters()

        self.test_weights = test_weights


    def eval(self):

        print("Resuming From ", self.test_weights)
        checkpoint = torch.load(self.test_weights)
        self.amodel.load_state_dict(checkpoint['state_dict'])

        range_image_paths_database = load_files(self.data_root_folder)
        print("scan number of database: ", len(range_image_paths_database))

        des_list = np.zeros((12500, 256 ))        # for forward driving
        for j in tqdm(range(0, 12500)):
            f1_index = str(j).zfill(6)
            current_batch = read_one_need_from_seq(self.data_root_folder, f1_index)
            current_batch_double = torch.cat((current_batch, current_batch), dim=-1)
            current_batch_inv = current_batch_double[:,:,:,450:1350]
            current_batch = torch.cat((current_batch, current_batch_inv), dim=0)
            self.amodel.eval()
            current_batch_des = self.amodel(current_batch)
            des_list[(j), :] = current_batch_des[0, :].cpu().detach().numpy()


        des_list = des_list.astype('float32')

        row_list = []

        range_image_paths_query = load_files(self.data_root_folder_test)
        print("scan number of query: ", len(range_image_paths_query))

        for i in range(0,13500, 5):
            nlist = 1
            k = 35
            d = 256
            quantizer = faiss.IndexFlatL2(d)

            index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
            assert not index.is_trained

            index.train(des_list)
            assert index.is_trained
            index.add(des_list)

            i_index = str(i).zfill(6)
            current_batch = read_one_need_from_seq_test(self.data_root_folder_test, i_index)   # compute 1 descriptors
            current_batch_double = torch.cat((current_batch, current_batch), dim=-1)
            current_batch_inv = current_batch_double[:,:,:,450:1350]
            current_batch = torch.cat((current_batch, current_batch_inv), dim=0)
            self.amodel.eval()
            current_batch_des = self.amodel(current_batch)   #  torch.Size([(1+pos_num+neg_num)), 256])
            des_list_current = current_batch_des[0, :].cpu().detach().numpy()
            D, I = index.search(des_list_current.reshape(1, -1), k)  # actual search

            for j in range(D.shape[1]):
                one_row = np.zeros((1,3))
                one_row[:, 0] = i
                one_row[:, 1] = I[:,j]
                one_row[:, 2] = D[:,j]
                row_list.append(one_row)
                print("02:"+str(i) + "---->" + "01:" + str(I[:, j]) + "  " + str(D[:, j]))

        row_list_arr = np.array(row_list)
        np.savez_compressed("./test_results_haomo/predicted_des_L2_dis_bet_traj_forward", row_list_arr)


if __name__ == '__main__':
    # data
    # load config ================================================================
    config_filename = '../config/config_haomo.yml'
    config = yaml.safe_load(open(config_filename))
    data_root_folder = config["file_root"]["data_root_folder"]
    data_root_folder_test = config["file_root"]["data_root_folder_test"]
    test_weights = config["file_root"]["test_weights"]
    # ============================================================================

    test_handler = testHandler(height=32, width=900, channels=1, norm_layer=None,use_transformer=True,
                                 data_root_folder=data_root_folder, data_root_folder_test=data_root_folder_test, test_weights=test_weights)

    test_handler.eval()
