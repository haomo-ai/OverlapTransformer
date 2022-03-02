#!/usr/bin/env python3
# Developed by Junyi Ma, Xieyuanli Chen, and Jun Zhang
# This file is covered by the LICENSE file in the root of the project OverlapTransformer:
# https://github.com/haomo-ai/OverlapTransformer/
# Brief: OverlapTransformer modules for Haomo dataset


import os
import sys
p = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
if p not in sys.path:
    sys.path.append(p)
sys.path.append('../tools/')    
import torch
import torch.nn as nn
import numpy as np

from modules.netvlad import NetVLADLoupe
import torch.nn.functional as F



class featureExtracter(nn.Module):
    def __init__(self, height=64, width=900, channels=5, norm_layer=None, use_transformer = True):
        super(featureExtracter, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d   # number of channels

        self.use_transformer = use_transformer

        self.conv1 = nn.Conv2d(channels, 16, kernel_size=(2,1), stride=(2,1), bias=False)
        self.bn1 = norm_layer(16)
        self.conv1_add = nn.Conv2d(16, 16, kernel_size=(5,1), stride=(1,1), bias=False)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3,1), stride=(1,1), bias=False)
        self.bn2 = norm_layer(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3,1), stride=(1,1), bias=False)
        self.bn3 = norm_layer(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=(3,1), stride=(1,1), bias=False)
        self.bn4 = norm_layer(64)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=(3,1), stride=(1,1), bias=False)
        self.bn5 = norm_layer(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=(3,1), stride=(1,1), bias=False)
        self.bn6 = norm_layer(128)
        self.conv7 = nn.Conv2d(128, 128, kernel_size=(1,1), stride=(2,1), bias=False)
        self.bn7 = norm_layer(128)

        self.relu = nn.ReLU(inplace=True)


        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=4, dim_feedforward=1024, activation='relu', batch_first=False,dropout=0.)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=1)  # 3 6
        self.convLast1 = nn.Conv2d(128, 256, kernel_size=(1,1), stride=(1,1), bias=False)
        self.bnLast1 = norm_layer(256)
        self.convLast2 = nn.Conv2d(512, 1024, kernel_size=(1,1), stride=(1,1), bias=False)
        self.bnLast2 = norm_layer(1024)

        self.linear = nn.Linear(128*900, 256)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

        self.net_vlad = NetVLADLoupe(feature_size=1024, max_samples=900, cluster_size=64,  # before 11.12 --- 64
                                     output_dim=256, gating=True, add_batch_norm=False,   # output_dim=512
                                     is_training=True)

        self.linear1 = nn.Linear(1 * 256, 256)
        self.bnl1 = norm_layer(256)
        self.linear2 = nn.Linear(1 * 256, 256)
        self.bnl2 = norm_layer(256)
        self.linear3 = nn.Linear(1 * 256, 256)
        self.bnl3 = norm_layer(256)

    def forward(self, x_l):


        out_l = self.relu(self.conv1(x_l))
        out_l = self.relu(self.conv1_add(out_l))
        out_l = self.relu(self.conv2(out_l))
        out_l = self.relu(self.conv3(out_l))
        out_l = self.relu(self.conv4(out_l))
        out_l = self.relu(self.conv5(out_l))
        out_l = self.relu(self.conv6(out_l))
        out_l = self.relu(self.conv7(out_l))


        out_l_1 = out_l.permute(0,1,3,2)  # out_r (bs, 128,360, 1)
        out_l_1 = self.relu(self.convLast1(out_l_1))

        if self.use_transformer:
            out_l = out_l_1.squeeze(3)

            out_l = out_l.permute(2, 0, 1)
            out_l = self.transformer_encoder(out_l)

            out_l = out_l.permute(1, 2, 0)
            out_l = out_l.unsqueeze(3)
            out_l = torch.cat((out_l_1, out_l), dim=1)
            out_l = self.relu(self.convLast2(out_l))
            out_l = F.normalize(out_l, dim=1)
            out_l = self.net_vlad(out_l)
            out_l = F.normalize(out_l, dim=1)

        return out_l

#
# if __name__ == '__main__':
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     # fig_traj2 = np.load("/home/mjy/datasets/haomo_data/1208_1/02/img_all/depth/002465.npy")
#     # fig_traj1 = np.load("/home/mjy/datasets/haomo_data/1208_1/01/img_all/depth/031291.npy")
#
#     # combined_tensor = read_one_need_from_seq("031291")   # torch.Size([1, 1, 32, 900])
#     # combined_tensor = read_one_need_from_seq_test("002465")   # torch.Size([1, 1, 32, 900])
#     # print(combined_tensor.size())
#
#     # toy example
#     combined_tensor = torch.zeros((1, 1, 32, 900)).to(device)
#     for i in range(5):
#         combined_tensor[:,:,:,180*i:180*(i+1)] = torch.ones((1, 1, 32, 180)) * 10 * i
#
#
#     current_batch_double = torch.cat((combined_tensor, combined_tensor), dim=-1)
#     current_batch_inv = current_batch_double[:, :, :, 450:1350]
#     # current_batch_inv = current_batch_double[:, :, :, 225:1125]
#
#     combined_tensor = torch.cat((combined_tensor, current_batch_inv), dim=0)
#
#     feature_extracter=featureExtracter(use_transformer=True,channels=1)
#     feature_extracter.to(device)
#
#     resume_filename = "/home/mjy/datasets/haomo_data/tools/amodel_transformer_depth_only13.pth.tar"
#     print("Resuming From ", resume_filename)
#     checkpoint = torch.load(resume_filename)
#     starting_epoch = checkpoint['epoch']
#     feature_extracter.load_state_dict(checkpoint['state_dict'])  # 加载状态字典
#
#
#     feature_extracter.eval()
#     output_l, output_l_1= feature_extracter(combined_tensor)
#     # print(output_l)
#     print(output_l.shape)   # torch.Size([2, 256])
#     print(output_l_1.shape)   # torch.Size([2, 256])
#
#     show_input_results(combined_tensor[0,:,:,:])
#     show_input_results(combined_tensor[1,:,:,:])
#
#     show_inter_results(output_l_1[0,:,:,:])
#     show_inter_results(output_l_1[1,:,:,:])
#
#     show_final_results(output_l[0,:])
#     show_final_results(output_l[1,:])
