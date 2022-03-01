import os
import sys
p = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
if p not in sys.path:
    sys.path.append(p)
    
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class NetVLADLoupe(nn.Module):
    def __init__(self, feature_size, max_samples, cluster_size, output_dim,
                 gating=True, add_batch_norm=True, is_training=True):
        super(NetVLADLoupe, self).__init__()
        self.feature_size = feature_size
        self.max_samples = max_samples
        self.output_dim = output_dim
        self.is_training = is_training
        self.gating = gating
        self.add_batch_norm = add_batch_norm
        self.cluster_size = cluster_size
        self.softmax = nn.Softmax(dim=-1)

        self.cluster_weights = nn.Parameter(torch.randn(
            feature_size, cluster_size) * 1 / math.sqrt(feature_size))
        self.cluster_weights2 = nn.Parameter(torch.randn(
            1, feature_size, cluster_size) * 1 / math.sqrt(feature_size))
        self.hidden1_weights = nn.Parameter(torch.randn(
            cluster_size * feature_size, output_dim) * 1 / math.sqrt(feature_size))

        if add_batch_norm:
            self.cluster_biases = None
            self.bn1 = nn.BatchNorm1d(cluster_size)
        else:
            self.cluster_biases = nn.Parameter(torch.randn(
                cluster_size) * 1 / math.sqrt(feature_size))
            self.bn1 = None

        self.bn2 = nn.BatchNorm1d(output_dim)

        if gating:
            self.context_gating = GatingContext(
                output_dim, add_batch_norm=add_batch_norm)

    def forward(self, x):
        x = x.transpose(1, 3).contiguous()
        x = x.view((-1, self.max_samples, self.feature_size))
        activation = torch.matmul(x, self.cluster_weights)
        if self.add_batch_norm:
            # activation = activation.transpose(1,2).contiguous()
            activation = activation.view(-1, self.cluster_size)
            activation = self.bn1(activation)
            activation = activation.view(-1, self.max_samples, self.cluster_size)
            # activation = activation.transpose(1,2).contiguous()
        else:
            activation = activation + self.cluster_biases
        activation = self.softmax(activation)
        activation = activation.view((-1, self.max_samples, self.cluster_size))

        a_sum = activation.sum(-2, keepdim=True)
        a = a_sum * self.cluster_weights2

        activation = torch.transpose(activation, 2, 1)
        x = x.view((-1, self.max_samples, self.feature_size))
        vlad = torch.matmul(activation, x)
        vlad = torch.transpose(vlad, 2, 1)
        vlad = vlad - a

        vlad = F.normalize(vlad, dim=1, p=2)
        # vlad = vlad.view((-1, self.cluster_size * self.feature_size))
        vlad = vlad.reshape((-1, self.cluster_size * self.feature_size))
        vlad = F.normalize(vlad, dim=1, p=2)

        vlad = torch.matmul(vlad, self.hidden1_weights)

        # vlad = self.bn2(vlad)

        if self.gating:
            vlad = self.context_gating(vlad)

        return vlad


class GatingContext(nn.Module):
    def __init__(self, dim, add_batch_norm=True):
        super(GatingContext, self).__init__()
        self.dim = dim
        self.add_batch_norm = add_batch_norm
        self.gating_weights = nn.Parameter(
            torch.randn(dim, dim) * 1 / math.sqrt(dim))
        self.sigmoid = nn.Sigmoid()

        if add_batch_norm:
            self.gating_biases = None
            self.bn1 = nn.BatchNorm1d(dim)
        else:
            self.gating_biases = nn.Parameter(
                torch.randn(dim) * 1 / math.sqrt(dim))
            self.bn1 = None

    def forward(self, x):
        gates = torch.matmul(x, self.gating_weights)

        if self.add_batch_norm:
            gates = self.bn1(gates)
        else:
            gates = gates + self.gating_biases

        gates = self.sigmoid(gates)

        activation = x * gates

        return activation


if __name__ == '__main__':
    net_vlad = NetVLADLoupe(feature_size=1024, max_samples=360, cluster_size=16,
                                 output_dim=20, gating=True, add_batch_norm=True,
                                 is_training=True)
    # input  (bs, 1024, 360, 1)
    torch.manual_seed(1234)
    input_tensor = F.normalize(torch.randn((1,1024,360,1)), dim=1)
    input_tensor2 = torch.zeros_like(input_tensor)
    input_tensor2[:, :, 2:, :] = input_tensor[:, :, 0:-2, :].clone()
    input_tensor2[:, :, :2, :]  = input_tensor[:, :, -2:, :].clone()
    input_tensor2= F.normalize(input_tensor2, dim=1)
    input_tensor_com = torch.cat((input_tensor, input_tensor2), dim=0)

    # print(input_tensor[0,0,:,0])
    # print(input_tensor2[0,0,:,0])
    print("==================================")

    with torch.no_grad():
        net_vlad.eval()
        # output_tensor = net_vlad(input_tensor_com)
        # print(output_tensor)
        out1 = net_vlad(input_tensor)
        print(out1)
        net_vlad.eval()
        # input_tensor2[:, :, 20:, :] = 0.1
        input_tensor2 = F.normalize(input_tensor2, dim=1)
        out2 = net_vlad(input_tensor2)
        print(out2)
        net_vlad.eval()
        input_tensor3 = torch.randn((1,1024,360,1))
        out3 = net_vlad(input_tensor3)
        print(out3)


        print(((out1-out2)**2).sum(1))
        print(((out1-out3)**2).sum(1))


