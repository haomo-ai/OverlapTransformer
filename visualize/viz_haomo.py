#!/usr/bin/env python3
# Developed by Junyi Ma, Xieyuanli Chen, and Jun Zhang
# This file is covered by the LICENSE file in the root of the project OverlapTransformer:
# https://github.com/haomo-ai/OverlapTransformer/
# Brief: Visualize evaluation on Hamo dataset


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
import os



class vizHandler():
    def __init__(self, height=32, width=900, channels=1, norm_layer=None,use_transformer=True,
                 data_root_folder=None, data_root_folder_test=None, pose_file_database=None, pose_file_query=None, test_weights=None):
        super(vizHandler, self).__init__()

        self.height = height
        self.width = width
        self.channels = channels
        self.norm_layer = norm_layer
        self.use_transformer = use_transformer
        self.data_root_folder = data_root_folder
        self.data_root_folder_test = data_root_folder_test
        self.pose_file_database = pose_file_database
        self.pose_file_query = pose_file_query
        self.poses_database = np.load(pose_file_database)
        self.poses_query = np.load(pose_file_query)

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

        if not os.path.exists("./des_list.npy"):
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
            np.save("des_list", des_list)
        else:
            des_list = np.load("des_list.npy")
            des_list = des_list.astype('float32')

        row_list = []

        range_image_paths_query = load_files(self.data_root_folder_test)
        print("scan number of query: ", len(range_image_paths_query))
        plt.figure(figsize=(15,5))
        plt.ion()
        for i in range(1000, 13500, 5):
            nlist = 1
            k = 1
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
            current_batch_all = torch.cat((current_batch, current_batch_inv), dim=0)
            self.amodel.eval()
            current_batch_des = self.amodel(current_batch_all)   #  torch.Size([(1+pos_num+neg_num)), 256])
            des_list_current = current_batch_des[0, :].cpu().detach().numpy()
            D, I = index.search(des_list_current.reshape(1, -1), k)  # actual search

            one_row = np.zeros((1,3))
            one_row[:, 0] = i
            one_row[:, 1] = I[:,0]
            one_row[:, 2] = D[:,0]
            row_list.append(one_row)
            print("02:"+str(i) + "---->" + "01:" + str(I[:, 0]) + "  " + str(D[:, 0]))

            plt.subplot(121)
            plt.scatter(self.poses_database[0,1], self.poses_database[0,2], s=1, c="blue", label="database")
            plt.scatter(self.poses_database[:,1], self.poses_database[:,2], s=0.05, c="blue")
            plt.scatter(self.poses_query[i,1], self.poses_query[i,2], linewidths=3, c="black", label="query")
            plt.scatter(self.poses_database[int(one_row[0, 1]),1], self.poses_database[int(one_row[0, 1]),2], linewidths=1, c="red", label="top1 candidate")
            plt.legend()
            plt.subplot(422)
            plt.title("range image (query)")
            plt.imshow(current_batch.cpu().detach().numpy()[0, 0, :, :])
            plt.subplot(426)
            plt.title("range image (database)")
            reference_batch = read_one_need_from_seq(self.data_root_folder, str(int(I[:,0])).zfill(6))
            plt.imshow(reference_batch.cpu().detach().numpy()[0, 0, :, :])
            plt.subplot(424)
            plt.title("global descriptor (query)")
            des_list_current_for_show = np.expand_dims(des_list_current, 0)
            des_list_current_for_show = np.repeat(des_list_current_for_show,10,0)
            plt.imshow(des_list_current_for_show)
            plt.subplot(428)
            plt.title("global descriptor (database)")
            des_list_reference = des_list[int(I[:,0]), :]
            des_list_reference_for_show = np.expand_dims(des_list_reference, 0)
            des_list_reference_for_show = np.repeat(des_list_reference_for_show,10,0)
            plt.imshow(des_list_reference_for_show)

            plt.pause(0.01)
            plt.clf()



if __name__ == '__main__':
    # data
    # load config ================================================================
    config_filename = '../config/config_haomo.yml'
    config = yaml.safe_load(open(config_filename))
    data_root_folder = config["file_root"]["data_root_folder"]
    data_root_folder_test = config["file_root"]["data_root_folder_test"]
    test_weights = config["file_root"]["test_weights"]
    pose_file_database = config["file_root"]["pose_file_database"]
    pose_file_query = config["file_root"]["pose_file_query"]
    # ============================================================================

    viz_handler = vizHandler(height=32, width=900, channels=1, norm_layer=None,use_transformer=True,
                             data_root_folder=data_root_folder, data_root_folder_test=data_root_folder_test,
                             pose_file_database=pose_file_database,
                             pose_file_query=pose_file_query,
                             test_weights=test_weights)

    viz_handler.eval()
