#!/usr/bin/env python3
# Developed by Junyi Ma, Xieyuanli Chen, and Jun Zhang
# This file is covered by the LICENSE file in the root of the project OverlapTransformer:
# https://github.com/haomo-ai/OverlapTransformer/
# Brief: OverlapTransformer modules for KITTI sequences


import os
import sys
p = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
if p not in sys.path:
    sys.path.append(p)
sys.path.append('../tools/')
import torch
import torch.nn as nn

from modules.netvlad import NetVLADLoupe
import torch.nn.functional as F
from tools.read_samples import read_one_need_from_seq
import yaml

"""
    Feature extracter of OverlapTransformer.
    Args:
        height: the height of the range image (64 for KITTI sequences). 
                 This is an interface for other types LIDAR.
        width: the width of the range image (900, alone the lines of OverlapNet).
                This is an interface for other types LIDAR.
        channels: 1 for depth only in our work. 
                This is an interface for multiple cues.
        norm_layer: None in our work for better model.
        use_transformer: Whether to use MHSA.
"""
class featureExtracter(nn.Module):
    def __init__(self, height=64, width=900, channels=5, norm_layer=None, use_transformer = True):
        super(featureExtracter, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.use_transformer = use_transformer

        self.conv1 = nn.Conv2d(channels, 16, kernel_size=(5,1), stride=(1,1), bias=False)
        self.bn1 = norm_layer(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3,1), stride=(2,1), bias=False)
        self.bn2 = norm_layer(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3,1), stride=(2,1), bias=False)
        self.bn3 = norm_layer(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=(3,1), stride=(2,1), bias=False)
        self.bn4 = norm_layer(64)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=(2,1), stride=(2,1), bias=False)
        self.bn5 = norm_layer(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=(1,1), stride=(2,1), bias=False)
        self.bn6 = norm_layer(128)
        self.conv7 = nn.Conv2d(128, 128, kernel_size=(1,1), stride=(2,1), bias=False)
        self.bn7 = norm_layer(128)
        self.conv8 = nn.Conv2d(128, 128, kernel_size=(1,1), stride=(2,1), bias=False)
        self.bn8 = norm_layer(128)
        self.conv9 = nn.Conv2d(128, 128, kernel_size=(1,1), stride=(2,1), bias=False)
        self.bn9 = norm_layer(128)
        self.conv10 = nn.Conv2d(128, 128, kernel_size=(1,1), stride=(2,1), bias=False)
        self.bn10 = norm_layer(128)
        self.conv11 = nn.Conv2d(128, 128, kernel_size=(1,1), stride=(2,1), bias=False)
        self.bn11 = norm_layer(128)
        self.relu = nn.ReLU(inplace=True)

        """
            MHSA
            num_layers=1 is suggested in our work.
        """
        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=4, dim_feedforward=1024, activation='relu', batch_first=False,dropout=0.)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.convLast1 = nn.Conv2d(128, 256, kernel_size=(1,1), stride=(1,1), bias=False)
        self.bnLast1 = norm_layer(256)
        self.convLast2 = nn.Conv2d(512, 1024, kernel_size=(1,1), stride=(1,1), bias=False)
        self.bnLast2 = norm_layer(1024)

        self.linear = nn.Linear(128*900, 256)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

        """
            NETVLAD
            add_batch_norm=False is needed in our work.
        """
        self.net_vlad = NetVLADLoupe(feature_size=1024, max_samples=900, cluster_size=64,
                                     output_dim=256, gating=True, add_batch_norm=False,
                                     is_training=True)

        """TODO: How about adding some dense layers?"""
        self.linear1 = nn.Linear(1 * 256, 256)
        self.bnl1 = norm_layer(256)
        self.linear2 = nn.Linear(1 * 256, 256)
        self.bnl2 = norm_layer(256)
        self.linear3 = nn.Linear(1 * 256, 256)
        self.bnl3 = norm_layer(256)

    def forward(self, x_l):

        out_l = self.relu(self.conv1(x_l))
        out_l = self.relu(self.conv2(out_l))
        out_l = self.relu(self.conv3(out_l))
        out_l = self.relu(self.conv4(out_l))
        out_l = self.relu(self.conv5(out_l))
        out_l = self.relu(self.conv6(out_l))
        out_l = self.relu(self.conv7(out_l))
        out_l = self.relu(self.conv8(out_l))
        out_l = self.relu(self.conv9(out_l))
        out_l = self.relu(self.conv10(out_l))
        out_l = self.relu(self.conv11(out_l))


        out_l_1 = out_l.permute(0,1,3,2)
        out_l_1 = self.relu(self.convLast1(out_l_1))

        """Using transformer needs to decide whether batch_size first"""
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

        else:
            out_l = torch.cat((out_l_1, out_l_1), dim=1)
            out_l = F.normalize(out_l, dim=1)
            out_l = self.net_vlad(out_l)
            out_l = F.normalize(out_l, dim=1)

        return out_l


if __name__ == '__main__':
    # load config ================================================================
    config_filename = '../config/config.yml'
    config = yaml.safe_load(open(config_filename))
    seqs_root = config["data_root"]["data_root_folder"]
    # ============================================================================

    combined_tensor = read_one_need_from_seq(seqs_root, "000000","00")
    combined_tensor = torch.cat((combined_tensor,combined_tensor), dim=0)

    feature_extracter=featureExtracter(use_transformer=True, channels=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extracter.to(device)
    feature_extracter.eval()

    print("model architecture: \n")
    print(feature_extracter)

    gloabal_descriptor = feature_extracter(combined_tensor)
    print("size of gloabal descriptor: \n")
    print(gloabal_descriptor.size())
