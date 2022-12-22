import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.resnet as resnet
from torch.nn import Module, Conv2d, Parameter, Softmax


def l2_norm(x):
    return torch.einsum("bcn, bn->bcn", x, 1 / torch.norm(x, p=2, dim=-2))


def softplus_feature_map(x):
    return torch.nn.functional.softplus(x)


class PositionLinearAttention(Module):
    def __init__(self, in_places, eps=1e-6):
        super(PositionLinearAttention, self).__init__()
        self.gamma = Parameter(torch.zeros(1))
        self.in_places = in_places
        self.l2_norm = l2_norm
        self.eps = eps

        self.query_conv = Conv2d(in_channels=in_places, out_channels=in_places // 8, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_places, out_channels=in_places // 8, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_places, out_channels=in_places, kernel_size=1)

    def forward(self, x):
        # Apply the feature map to the queries and keys
        batch_size, chnnels, width, height = x.shape
        Q = self.query_conv(x).view(batch_size, -1, width * height)
        K = self.key_conv(x).view(batch_size, -1, width * height)
        V = self.value_conv(x).view(batch_size, -1, width * height)

        Q = self.l2_norm(Q).permute(-3, -1, -2)
        K = self.l2_norm(K)

        tailor_sum = 1 / (width * height + torch.einsum("bnc, bc->bn", Q, torch.sum(K, dim=-1) + self.eps))
        value_sum = torch.einsum("bcn->bc", V).unsqueeze(-1)
        value_sum = value_sum.expand(-1, chnnels, width * height)
        matrix = torch.einsum('bcn, bnm->bcm', V, Q)
        matrix_sum = value_sum + torch.einsum("bcm, bmn->bcn", matrix, K)

        weight_value = torch.einsum("bcn, bn->bcn", matrix_sum, tailor_sum)
        weight_value = weight_value.view(batch_size, chnnels, height, width)

        return (x + self.gamma * weight_value).contiguous()


class ChannelLinearAttention(Module):
    def __init__(self):
        super(ChannelLinearAttention, self).__init__()
        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        batch_size, chnnels, width, height = x.shape
        proj_query = x.view(batch_size, chnnels, -1)
        proj_key = x.view(batch_size, chnnels, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(batch_size, chnnels, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(batch_size, chnnels, height, width)

        out = self.gamma * out + x
        return out