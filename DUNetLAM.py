# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 01:49:28 2021

@author: TALHA
"""

# from torch import nn
import torch.nn as nn
import torch
from attention import PositionLinearAttention, ChannelLinearAttention
#from models.DUpsample import DUpsampling


def conv3otherRelu(in_planes, out_planes, kernel_size=None, stride=None, padding=None):
    # 3x3 convolution with padding and relu
    if kernel_size is None:
        kernel_size = 3
    assert isinstance(kernel_size, (int, tuple)), 'kernel_size is not in (int, tuple)!'

    if stride is None:
        stride = 1
    assert isinstance(stride, (int, tuple)), 'stride is not in (int, tuple)!'

    if padding is None:
        padding = 1
    assert isinstance(padding, (int, tuple)), 'padding is not in (int, tuple)!'

    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
        # nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True)  # inplace=True
    )

class DUpsampling(nn.Module):
    def __init__(self, inplanes, scale_factor, class_num, pad=0):
        super(DUpsampling, self).__init__()
        ## W matrix
        self.conv_w = nn.Conv2d(inplanes, class_num * scale_factor * scale_factor, kernel_size=1, padding = pad,bias=False)
        ## P matrix
        self.conv_p = nn.Conv2d(class_num * scale_factor * scale_factor, inplanes*scale_factor, kernel_size=1, padding = pad,bias=False)

        self.scale = scale_factor
    
    def forward(self, x):
        x = self.conv_w(x)
        x = self.conv_p(x)
        N, C, H, W = x.size()

        # N, W, H, C
        x_permuted = x.permute(0, 3, 2, 1)

        # N, W, H*scale, C/scale
        x_permuted = x_permuted.contiguous().view((N, W, H * self.scale, int(C / (self.scale))))

        # N, H*scale, W, C/scale
        x_permuted = x_permuted.permute(0, 2, 1, 3)
        # N, H*scale, W*scale, C/(scale**2)
        x_permuted = x_permuted.contiguous().view((N, W * self.scale, H * self.scale, int(C / (self.scale * self.scale))))

        # N, C/(scale**2), H*scale, W*scale
        x = x_permuted.permute(0, 3, 1, 2)
        
        return x




class DUNetDenseLAMfour(nn.Module):
    def __init__(self, band_num, class_num):
        super(DUNetDenseLAMfour, self).__init__()
        self.band_num = band_num
        self.class_num = class_num
        self.name = 'DUNetDenseLAMfour-Modify'

        channels = [32, 64, 128, 256, 512]
        self.conv1 = nn.Sequential(
            conv3otherRelu(self.band_num, channels[0]),
            conv3otherRelu(channels[0], channels[0]),
        )

        self.conv2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            conv3otherRelu(channels[0], channels[1]),
            conv3otherRelu(channels[1], channels[1]),
        )

        self.conv3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            conv3otherRelu(channels[1], channels[2]),
            conv3otherRelu(channels[2], channels[2]),
            conv3otherRelu(channels[2], channels[2]),
        )

        self.conv4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            conv3otherRelu(channels[2], channels[3]),
            conv3otherRelu(channels[3], channels[3]),
            conv3otherRelu(channels[3], channels[3]),
        )

        self.conv5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            conv3otherRelu(channels[3], channels[4]),
            conv3otherRelu(channels[4], channels[4]),
            conv3otherRelu(channels[4], channels[4]),
        )

        self.deconv4 = DUpsampling(channels[4], scale_factor = 2, class_num=5)
        self.conv6 = nn.Sequential(
            conv3otherRelu(channels[4], channels[3]),
            conv3otherRelu(channels[3], channels[3]),
        )
        
        self.up_sample_Dense_11 = DUpsampling(
                                        channels[3],
                                        scale_factor = 2, 
                                        class_num=5)
        
        self.up_sample_Dense_12 = DUpsampling(
                                        channels[3],
                                        scale_factor = 4, 
                                        class_num=5)
        
        self.up_sample_Dense_13 = DUpsampling(
                                        channels[3],
                                        scale_factor = 8, 
                                        class_num=5)
        
        self.lpa4 = PositionLinearAttention(channels[4])
        self.lca4 = ChannelLinearAttention()

        self.deconv3 = DUpsampling(channels[3], scale_factor = 2, class_num=5)
        self.conv7 = nn.Sequential(
            conv3otherRelu(channels[3], channels[2]),
            conv3otherRelu(channels[2], channels[2]),
        )
        
        self.lpa3 = PositionLinearAttention(channels[3])
        self.lca3 = ChannelLinearAttention()
        
        self.up_sample_Dense_2 = DUpsampling(
                                        channels[2],
                                        scale_factor = 2, 
                                        class_num=5)
        
        self.up_sample_Dense_21 = DUpsampling(
                                        channels[2],
                                        scale_factor = 4, 
                                        class_num=5)
        
        

        self.deconv2 = DUpsampling(channels[2], scale_factor = 2, class_num=5)
        self.conv8 = nn.Sequential(
            conv3otherRelu(channels[2], channels[1], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            conv3otherRelu(channels[1], channels[1], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )
        self.conv81 = nn.Sequential(
            conv3otherRelu(channels[2]+64, channels[1], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            conv3otherRelu(channels[1], channels[1], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )
        
        self.lpa2 = PositionLinearAttention(channels[2]+64)
        self.lca2 = ChannelLinearAttention()
        
        self.up_sample_Dense_3 = DUpsampling(
                                        channels[1],
                                        scale_factor = 2, 
                                        class_num=5)

        self.deconv1 = DUpsampling(channels[1], scale_factor = 2, class_num=5)
        self.conv9 = nn.Sequential(
            conv3otherRelu(channels[1], channels[0], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            conv3otherRelu(channels[0], channels[0], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )
        
        self.lpa1 = PositionLinearAttention(channels[2])
        self.lca1 = ChannelLinearAttention()

        self.conv10 = nn.Conv2d(channels[2], self.class_num, kernel_size=1, stride=1)
        

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        # Decoder Block 1        
        deconv4 = self.deconv4(conv5)
        conv6 = torch.cat((deconv4, conv4), 1)
        
        lpa = self.lpa4(conv6)
        lca = self.lca4(conv6)
        feat_sum = lpa + lca
        conv6 = self.conv6(feat_sum)
 
        x11 = self.up_sample_Dense_11(conv6)
        x12 = self.up_sample_Dense_12(conv6)                               
        x13 = self.up_sample_Dense_13(conv6)                               

        # Decoder Block 2
        deconv3 = self.deconv3(conv6)
        conv7 = torch.cat((deconv3, conv3), 1)
        conv7 = self.conv7(conv7)
        conv7 = torch.cat([conv7, x11], 1)
        
        lpa = self.lpa3(conv7)
        lca = self.lca3(conv7)
        feat_sum = lpa + lca
        conv7 = self.conv7(feat_sum)
       
        x20 = self.up_sample_Dense_2(conv7)
        x21 = self.up_sample_Dense_21(conv7)                               
              
        # Decoder Block 3
        deconv2 = self.deconv2(conv7)
       
        conv8 = torch.cat((deconv2, conv2), 1)
        conv8 = self.conv8(conv8)
        conv8 = torch.cat([conv8, x12, x20], 1)
        
        lpa = self.lpa2(conv8)
        lca = self.lca2(conv8)
        feat_sum = lpa + lca
        conv8 = self.conv81(feat_sum)
        
        x30 = self.up_sample_Dense_3(conv8)

        # Decoder Block 4 
        deconv1 = self.deconv1(conv8)
        conv9 = torch.cat((deconv1, conv1), 1)
        conv9 = self.conv9(conv9)

        conv9 = torch.cat([conv9, x13, x21, x30], 1)

        
        lpa = self.lpa1(conv9)
        lca = self.lca1(conv9)
        feat_sum = lpa + lca

        output = self.conv10(feat_sum)

        return output