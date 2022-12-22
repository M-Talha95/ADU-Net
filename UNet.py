# from torch import nn
import torch.nn as nn
import torch
from models.LAMNet import PositionLinearAttention, ChannelLinearAttention, LinearAttention
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
        self.conv_p = nn.Conv2d(class_num * scale_factor * scale_factor, inplanes* scale_factor, kernel_size=1, padding = pad,bias=False)

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

def pre_compute_W(self, i, data):
        self.model.zero_grad()
        self.seggt = data[1]
        N, H, W = self.seggt.size()
        C = self.opt.label_nc
        scale = self.model.decoder.dupsample.scale
        # N, C, H, W
        self.seggt = torch.unsqueeze(self.seggt, dim=1)

        self.seggt[self.seggt == 255] = 0
        self.seggt_onehot = torch.zeros(N, C, H, W).scatter_(1, self.seggt.long(), 1)
        # N, H, W, C
        self.seggt_onehot = self.seggt_onehot.permute(0, 2, 3, 1)
        # N, H, W/sacle, C*scale
        self.seggt_onehot = self.seggt_onehot.contiguous().view((N, H, 
                                        int(W / scale), C * scale))
        # N, W/sacle, H, C*scale
        self.seggt_onehot = self.seggt_onehot.permute(0, 2, 1, 3)

        self.seggt_onehot = self.seggt_onehot.contiguous().view((N, int(W / scale), 
                                        int(H / scale), C * scale * scale))

        self.seggt_onehot = self.seggt_onehot.permute(0, 3, 2, 1)

        self.seggt_onehot = self.seggt_onehot.cuda()

        self.seggt_onehot_reconstructed = self.model.decoder.dupsample.conv_w(
                                            self.model.decoder.dupsample.conv_p(self.seggt_onehot))
        self.reconstruct_loss = torch.mean(torch.pow(self.seggt_onehot -
                                            self.seggt_onehot_reconstructed, 2))
        self.reconstruct_loss.backward()
        self.optimizer_w.step()
        if i % 20 == 0:
            print ('pre_compute_loss: %f' % (self.reconstruct_loss.data[0]))

class UNetLAM(nn.Module):
    def __init__(self, band_num, class_num):
        super(UNetLAM, self).__init__()
        self.band_num = band_num
        self.class_num = class_num
        self.name = 'UNetLAM'

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

        self.deconv4 = nn.ConvTranspose2d(channels[4], channels[3], kernel_size=(2, 2), stride=(2, 2))
        self.conv6 = nn.Sequential(
            conv3otherRelu(channels[4], channels[3]),
            conv3otherRelu(channels[3], channels[3]),
        )

        self.deconv3 = nn.ConvTranspose2d(channels[3], channels[2], kernel_size=(2, 2), stride=(2, 2))
        self.conv7 = nn.Sequential(
            conv3otherRelu(channels[3], channels[2]),
            conv3otherRelu(channels[2], channels[2]),
        )

        self.deconv2 = nn.ConvTranspose2d(channels[2], channels[1], kernel_size=(2, 2), stride=(2, 2))
        self.conv8 = nn.Sequential(
            conv3otherRelu(channels[2], channels[1], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            conv3otherRelu(channels[1], channels[1], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )

        self.deconv1 = nn.ConvTranspose2d(channels[1], channels[0], kernel_size=(2, 2), stride=(2, 2))
        self.conv9 = nn.Sequential(
            conv3otherRelu(channels[1], channels[0], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            conv3otherRelu(channels[0], channels[0], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )

        self.lpa = PositionLinearAttention(channels[0])
        self.lca = ChannelLinearAttention()

        self.conv10 = nn.Conv2d(channels[0], self.class_num, kernel_size=1, stride=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        deconv4 = self.deconv4(conv5)
        conv6 = torch.cat((deconv4, conv4), 1)
        conv6 = self.conv6(conv6)

        deconv3 = self.deconv3(conv6)
        conv7 = torch.cat((deconv3, conv3), 1)
        conv7 = self.conv7(conv7)

        deconv2 = self.deconv2(conv7)
        conv8 = torch.cat((deconv2, conv2), 1)
        conv8 = self.conv8(conv8)

        deconv1 = self.deconv1(conv8)
        conv9 = torch.cat((deconv1, conv1), 1)
        conv9 = self.conv9(conv9)

        lpa = self.lpa(conv9)

        lca = self.lca(conv9)

        feat_sum = lpa + lca

        output = self.conv10(feat_sum)

        return output


class UNetLinear(nn.Module):
    def __init__(self, band_num, class_num):
        super(UNetLinear, self).__init__()
        self.band_num = band_num
        self.class_num = class_num
        self.name = 'UNetLinear'

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

        self.deconv4 = nn.ConvTranspose2d(channels[4], channels[3], kernel_size=(2, 2), stride=(2, 2))
        self.conv6 = nn.Sequential(
            conv3otherRelu(channels[4], channels[3]),
            conv3otherRelu(channels[3], channels[3]),
        )

        self.deconv3 = nn.ConvTranspose2d(channels[3], channels[2], kernel_size=(2, 2), stride=(2, 2))
        self.conv7 = nn.Sequential(
            conv3otherRelu(channels[3], channels[2]),
            conv3otherRelu(channels[2], channels[2]),
        )

        self.deconv2 = nn.ConvTranspose2d(channels[2], channels[1], kernel_size=(2, 2), stride=(2, 2))
        self.conv8 = nn.Sequential(
            conv3otherRelu(channels[2], channels[1], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            conv3otherRelu(channels[1], channels[1], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )

        self.deconv1 = nn.ConvTranspose2d(channels[1], channels[0], kernel_size=(2, 2), stride=(2, 2))
        self.conv9 = nn.Sequential(
            conv3otherRelu(channels[1], channels[0], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            conv3otherRelu(channels[0], channels[0], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )

        self.lpa = LinearAttention(channels[0])
        self.lca = ChannelLinearAttention()

        self.conv10 = nn.Conv2d(channels[0], self.class_num, kernel_size=1, stride=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        deconv4 = self.deconv4(conv5)
        conv6 = torch.cat((deconv4, conv4), 1)
        conv6 = self.conv6(conv6)

        deconv3 = self.deconv3(conv6)
        conv7 = torch.cat((deconv3, conv3), 1)
        conv7 = self.conv7(conv7)

        deconv2 = self.deconv2(conv7)
        conv8 = torch.cat((deconv2, conv2), 1)
        conv8 = self.conv8(conv8)

        deconv1 = self.deconv1(conv8)
        conv9 = torch.cat((deconv1, conv1), 1)
        conv9 = self.conv9(conv9)

        lpa = self.lpa(conv9)

        lca = self.lca(conv9)

        feat_sum = lpa + lca

        output = self.conv10(feat_sum)

        return output


class UNet(nn.Module):
    def __init__(self, band_num, class_num):
        super(UNet, self).__init__()
        self.band_num = band_num
        self.class_num = class_num
        self.name = 'UNet'

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

        self.deconv4 = nn.ConvTranspose2d(channels[4], channels[3], kernel_size=(2, 2), stride=(2, 2))
        self.conv6 = nn.Sequential(
            conv3otherRelu(channels[4], channels[3]),
            conv3otherRelu(channels[3], channels[3]),
        )

        self.deconv3 = nn.ConvTranspose2d(channels[3], channels[2], kernel_size=(2, 2), stride=(2, 2))
        self.conv7 = nn.Sequential(
            conv3otherRelu(channels[3], channels[2]),
            conv3otherRelu(channels[2], channels[2]),
        )

        self.deconv2 = nn.ConvTranspose2d(channels[2], channels[1], kernel_size=(2, 2), stride=(2, 2))
        self.conv8 = nn.Sequential(
            conv3otherRelu(channels[2], channels[1], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            conv3otherRelu(channels[1], channels[1], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )

        self.deconv1 = nn.ConvTranspose2d(channels[1], channels[0], kernel_size=(2, 2), stride=(2, 2))
        self.conv9 = nn.Sequential(
            conv3otherRelu(channels[1], channels[0], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            conv3otherRelu(channels[0], channels[0], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )

        self.conv10 = nn.Conv2d(channels[0], self.class_num, kernel_size=1, stride=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        deconv4 = self.deconv4(conv5)
        conv6 = torch.cat((deconv4, conv4), 1)
        conv6 = self.conv6(conv6)

        deconv3 = self.deconv3(conv6)
        conv7 = torch.cat((deconv3, conv3), 1)
        conv7 = self.conv7(conv7)

        deconv2 = self.deconv2(conv7)
        conv8 = torch.cat((deconv2, conv2), 1)
        conv8 = self.conv8(conv8)

        deconv1 = self.deconv1(conv8)
        conv9 = torch.cat((deconv1, conv1), 1)
        conv9 = self.conv9(conv9)

        output = self.conv10(conv9)

        return output


class UNetTiny(nn.Module):
    def __init__(self, band_num, class_num):
        super(UNetTiny, self).__init__()
        self.band_num = band_num
        self.class_num = class_num
        self.name = 'UNetTiny'

        channels = [32, 64, 128, 256]
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

        self.deconv3 = nn.ConvTranspose2d(channels[3], channels[2], kernel_size=(2, 2), stride=(2, 2))
        self.conv7 = nn.Sequential(
            conv3otherRelu(channels[3], channels[2]),
            conv3otherRelu(channels[2], channels[2]),
        )

        self.deconv2 = nn.ConvTranspose2d(channels[2], channels[1], kernel_size=(2, 2), stride=(2, 2))
        self.conv8 = nn.Sequential(
            conv3otherRelu(channels[2], channels[1], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            conv3otherRelu(channels[1], channels[1], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )

        self.deconv1 = nn.ConvTranspose2d(channels[1], channels[0], kernel_size=(2, 2), stride=(2, 2))
        self.conv9 = nn.Sequential(
            conv3otherRelu(channels[1], channels[0], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            conv3otherRelu(channels[0], channels[0], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )

        self.conv10 = nn.Conv2d(channels[0], self.class_num, kernel_size=1, stride=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        deconv3 = self.deconv3(conv4)
        conv7 = torch.cat((deconv3, conv3), 1)
        conv7 = self.conv7(conv7)

        deconv2 = self.deconv2(conv7)
        conv8 = torch.cat((deconv2, conv2), 1)
        conv8 = self.conv8(conv8)

        deconv1 = self.deconv1(conv8)
        conv9 = torch.cat((deconv1, conv1), 1)
        conv9 = self.conv9(conv9)

        output = self.conv10(conv9)

        return output

class UNetBilinear(nn.Module):
    def __init__(self, band_num, class_num):
        super(UNetBilinear, self).__init__()
        self.band_num = band_num
        self.class_num = class_num
        self.name = 'UNetBilinear-Set'

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
            conv3otherRelu(channels[4], channels[4]//2),
        )

        self.deconv4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) #(channels[4], channels[3], kernel_size=(2, 2), stride=(2, 2))
        self.conv6 = nn.Sequential(
            conv3otherRelu(channels[4], channels[3]),
            conv3otherRelu(channels[3], channels[3]//2),
        )

        self.deconv3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  #(channels[3], channels[2], kernel_size=(2, 2), stride=(2, 2))
        self.conv7 = nn.Sequential(
            conv3otherRelu(channels[3], channels[2]),
            conv3otherRelu(channels[2], channels[2]//2),
        )

        self.deconv2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  #(channels[2], channels[1], kernel_size=(2, 2), stride=(2, 2))
        self.conv8 = nn.Sequential(
            conv3otherRelu(channels[2], channels[1], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            conv3otherRelu(channels[1], channels[1]//2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )

        self.deconv1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  #(channels[1], channels[0], kernel_size=(2, 2), stride=(2, 2))
        self.conv9 = nn.Sequential(
            conv3otherRelu(channels[1], channels[0], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            conv3otherRelu(channels[0], channels[0], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )

        self.conv10 = nn.Conv2d(channels[0], self.class_num, kernel_size=1, stride=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        deconv4 = self.deconv4(conv5)
        conv6 = torch.cat((deconv4, conv4), 1)
        conv6 = self.conv6(conv6)

        deconv3 = self.deconv3(conv6)
        conv7 = torch.cat((deconv3, conv3), 1)
        conv7 = self.conv7(conv7)

        deconv2 = self.deconv2(conv7)
        conv8 = torch.cat((deconv2, conv2), 1)
        conv8 = self.conv8(conv8)

        deconv1 = self.deconv1(conv8)
        conv9 = torch.cat((deconv1, conv1), 1)
        conv9 = self.conv9(conv9)

        output = self.conv10(conv9)

        return output

class PUNet(nn.Module):
    def __init__(self, band_num, class_num):
        super(PUNet, self).__init__()
        self.band_num = band_num
        self.class_num = class_num
        self.name = 'PUNet'

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

        self.deconv4 = nn.PixelShuffle(2) #(channels[4], channels[3], kernel_size=(2, 2), stride=(2, 2))
        self.conv6 = nn.Sequential(
            conv3otherRelu(channels[4]-128, channels[3]),
            conv3otherRelu(channels[3], channels[3]),
        )

        self.deconv3 = nn.PixelShuffle(2)  #(channels[3], channels[2], kernel_size=(2, 2), stride=(2, 2))
        self.conv7 = nn.Sequential(
            conv3otherRelu(channels[3]-64, channels[2]),
            conv3otherRelu(channels[2], channels[2]),
        )

        self.deconv2 = nn.PixelShuffle(2)  #(channels[2], channels[1], kernel_size=(2, 2), stride=(2, 2))
        self.conv8 = nn.Sequential(
            conv3otherRelu(channels[2]-32, channels[1], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            conv3otherRelu(channels[1], channels[1], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )

        self.deconv1 = nn.PixelShuffle(2)  #(channels[1], channels[0], kernel_size=(2, 2), stride=(2, 2))
        self.conv9 = nn.Sequential(
            conv3otherRelu(channels[1]-16, channels[0], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            conv3otherRelu(channels[0], channels[0], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )

        self.conv10 = nn.Conv2d(channels[0], self.class_num, kernel_size=1, stride=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        deconv4 = self.deconv4(conv5)
        conv6 = torch.cat((deconv4, conv4), 1)
        conv6 = self.conv6(conv6)

        deconv3 = self.deconv3(conv6)
        conv7 = torch.cat((deconv3, conv3), 1)
        conv7 = self.conv7(conv7)

        deconv2 = self.deconv2(conv7)
        conv8 = torch.cat((deconv2, conv2), 1)
        conv8 = self.conv8(conv8)

        deconv1 = self.deconv1(conv8)
        conv9 = torch.cat((deconv1, conv1), 1)
        conv9 = self.conv9(conv9)

        output = self.conv10(conv9)

        return output

class DUNet(nn.Module):    
    def __init__(self, band_num, class_num):
        super(DUNet, self).__init__()
        self.band_num = band_num
        self.class_num = class_num
        self.name = 'DUNet-Set-Again'

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

        self.deconv3 = DUpsampling(channels[3], scale_factor = 2, class_num=5)
        self.conv7 = nn.Sequential(
            conv3otherRelu(channels[3], channels[2]),
            conv3otherRelu(channels[2], channels[2]),
        )

        self.deconv2 = DUpsampling(channels[2], scale_factor = 2, class_num=5)
        self.conv8 = nn.Sequential(
            conv3otherRelu(channels[2], channels[1], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            conv3otherRelu(channels[1], channels[1], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )

        self.deconv1 = DUpsampling(channels[1], scale_factor = 2, class_num=5)
        self.conv9 = nn.Sequential(
            conv3otherRelu(channels[1], channels[0], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            conv3otherRelu(channels[0], channels[0], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )
        self.T = torch.nn.Parameter(torch.Tensor([1.00]))


        self.conv10 = nn.Conv2d(channels[0], self.class_num, kernel_size=1, stride=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        deconv4 = self.deconv4(conv5)
        deconv4 = deconv4 / self.T
        # print('size of DECONV4 :', deconv4.size())
        conv6 = torch.cat((deconv4, conv4), 1)
        conv6 = self.conv6(conv6)

        deconv3 = self.deconv3(conv6)
        deconv3 = deconv3 / self.T
        conv7 = torch.cat((deconv3, conv3), 1)
        conv7 = self.conv7(conv7)

        deconv2 = self.deconv2(conv7)
        deconv2 = deconv2 / self.T
        conv8 = torch.cat((deconv2, conv2), 1)
        conv8 = self.conv8(conv8)

        deconv1 = self.deconv1(conv8)
        deconv1 = deconv1 / self.T
        conv9 = torch.cat((deconv1, conv1), 1)
        conv9 = self.conv9(conv9)

        output = self.conv10(conv9)

        return output
    
class UNetDense(nn.Module):
    def __init__(self, band_num, class_num):
        super(UNetDense, self).__init__()
        self.band_num = band_num
        self.class_num = class_num
        self.name = 'UNetDense-5C-Again'

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

        self.deconv4 = nn.ConvTranspose2d(channels[4], channels[3], kernel_size=(2, 2), stride=(2, 2))
        self.conv6 = nn.Sequential(
            conv3otherRelu(channels[4], channels[3]),
            conv3otherRelu(channels[3], channels[3]),
        )
        
        self.up_sample_Dense_11 = nn.ConvTranspose2d(
                                        channels[3],
                                        channels[3],
                                        kernel_size=(2, 2),
                                        stride=2)
        
        self.up_sample_Dense_12 = nn.ConvTranspose2d(
                                        channels[3],
                                        channels[3],
                                        kernel_size=(4, 4),
                                        stride=4)
        
        self.up_sample_Dense_13 = nn.ConvTranspose2d(
                                        channels[3],
                                        channels[3],
                                        kernel_size=(8, 8),
                                        stride=8)

        self.deconv3 = nn.ConvTranspose2d(channels[3], channels[2], kernel_size=(2, 2), stride=(2, 2))
        self.conv7 = nn.Sequential(
            conv3otherRelu(channels[3], channels[2]),
            conv3otherRelu(channels[2], channels[2]),
        )
        
        self.up_sample_Dense_2 = nn.ConvTranspose2d(
                                        channels[2]+256,
                                        channels[2],
                                        kernel_size=(2, 2),
                                        stride=2)
        
        self.up_sample_Dense_21 = nn.ConvTranspose2d(
                                        channels[2]+256,
                                        channels[2],
                                        kernel_size=(4, 4),
                                        stride=4)

        self.deconv2 = nn.ConvTranspose2d(channels[2]+256, channels[1], kernel_size=(2, 2), stride=(2, 2))
        self.conv8 = nn.Sequential(
            conv3otherRelu(channels[2], channels[1], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            conv3otherRelu(channels[1], channels[1], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )
        
        self.up_sample_Dense_3 = nn.ConvTranspose2d(
                                        channels[1]+384,
                                        channels[1],
                                        kernel_size=(2, 2),
                                        stride=2)

        self.deconv1 = nn.ConvTranspose2d(channels[1]+384, channels[0], kernel_size=(2, 2), stride=(2, 2))
        self.conv9 = nn.Sequential(
            conv3otherRelu(channels[1], channels[0], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            conv3otherRelu(channels[0], channels[0], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )

        self.conv10 = nn.Conv2d(channels[0]+448, self.class_num, kernel_size=1, stride=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)


        # Decoder Block 1
        
        deconv4 = self.deconv4(conv5)
        conv6 = torch.cat((deconv4, conv4), 1)
        conv6 = self.conv6(conv6)
        
        x11 = self.up_sample_Dense_11(conv6)
#        print('size of X_Dense_11 :', x11.size())           #  ([1, 512, 128, 128])
        x12 = self.up_sample_Dense_12(conv6)                               
#        print('size of X_Dense_12 :', x12.size())           #  ([1, 512, 256, 256])
        x13 = self.up_sample_Dense_13(conv6)                               
#        print('size of X_Dense_13 :', x13.size())           #  ([1, 512, 512, 512])

        # Decoder Block 2
        deconv3 = self.deconv3(conv6)
        conv7 = torch.cat((deconv3, conv3), 1)
        conv7 = self.conv7(conv7)
        conv7 = torch.cat([conv7,x11],1)
        
        x20 = self.up_sample_Dense_2(conv7)
#        print('size of X_Dense_11 :', x11.size())           #  ([1, 512, 128, 128])
        x21 = self.up_sample_Dense_21(conv7)                               
#        print('size of X_Dense_12 :', x12.size())           #  ([1, 512, 256, 256])
        
        
        # Decoder Block 3
        deconv2 = self.deconv2(conv7)
        conv8 = torch.cat((deconv2, conv2), 1)
        conv8 = self.conv8(conv8)
        conv8 = torch.cat([conv8, x12, x20], 1)
        
        x30 = self.up_sample_Dense_3(conv8)

        # Decoder Block 4
        
        deconv1 = self.deconv1(conv8)
        conv9 = torch.cat((deconv1, conv1), 1)
        conv9 = self.conv9(conv9)
        conv9 = torch.cat([conv9, x13, x21, x30], 1)

        output = self.conv10(conv9)

        return output
    
class UNetDenseCM(nn.Module):
    def __init__(self, band_num, class_num):
        super(UNetDenseCM, self).__init__()
        self.band_num = band_num
        self.class_num = class_num
        self.name = 'UNetDenseCM'

        channels = [64, 128, 256, 512, 1024]
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

        self.deconv4 = nn.ConvTranspose2d(channels[4], channels[3], kernel_size=(2, 2), stride=(2, 2))
        self.conv6 = nn.Sequential(
            conv3otherRelu(channels[4], channels[3]),
            conv3otherRelu(channels[3], channels[3]),
        )
        
        self.up_sample_Dense_11 = nn.ConvTranspose2d(
                                        channels[3],
                                        channels[3],
                                        kernel_size=(2, 2),
                                        stride=2)
        
        self.up_sample_Dense_12 = nn.ConvTranspose2d(
                                        channels[3],
                                        channels[3],
                                        kernel_size=(4, 4),
                                        stride=4)
        
        self.up_sample_Dense_13 = nn.ConvTranspose2d(
                                        channels[3],
                                        channels[3],
                                        kernel_size=(8, 8),
                                        stride=8)

        self.deconv3 = nn.ConvTranspose2d(channels[3], channels[2], kernel_size=(2, 2), stride=(2, 2))
        self.conv7 = nn.Sequential(
            conv3otherRelu(channels[3], channels[2]),
            conv3otherRelu(channels[2], channels[2]),
        )
        
        self.up_sample_Dense_2 = nn.ConvTranspose2d(
                                        channels[2]+512,
                                        channels[2],
                                        kernel_size=(2, 2),
                                        stride=2)
        
        self.up_sample_Dense_21 = nn.ConvTranspose2d(
                                        channels[2]+512,
                                        channels[2],
                                        kernel_size=(4, 4),
                                        stride=4)

        self.deconv2 = nn.ConvTranspose2d(channels[2]+512, channels[1], kernel_size=(2, 2), stride=(2, 2))
        self.conv8 = nn.Sequential(
            conv3otherRelu(channels[2], channels[1], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            conv3otherRelu(channels[1], channels[1], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )
        
        self.up_sample_Dense_3 = nn.ConvTranspose2d(
                                        channels[1]+768,
                                        channels[1],
                                        kernel_size=(2, 2),
                                        stride=2)

        self.deconv1 = nn.ConvTranspose2d(channels[1]+768, channels[0], kernel_size=(2, 2), stride=(2, 2))
        self.conv9 = nn.Sequential(
            conv3otherRelu(channels[1], channels[0], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            conv3otherRelu(channels[0], channels[0], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )

        self.conv10 = nn.Conv2d(channels[0]+896, self.class_num, kernel_size=1, stride=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)


        # Decoder Block 1
        
        deconv4 = self.deconv4(conv5)
        conv6 = torch.cat((deconv4, conv4), 1)
        conv6 = self.conv6(conv6)
        
        x11 = self.up_sample_Dense_11(conv6)
#        print('size of X_Dense_11 :', x11.size())           #  ([1, 512, 128, 128])
        x12 = self.up_sample_Dense_12(conv6)                               
#        print('size of X_Dense_12 :', x12.size())           #  ([1, 512, 256, 256])
        x13 = self.up_sample_Dense_13(conv6)                               
#        print('size of X_Dense_13 :', x13.size())           #  ([1, 512, 512, 512])

        # Decoder Block 2
        deconv3 = self.deconv3(conv6)
        conv7 = torch.cat((deconv3, conv3), 1)
        conv7 = self.conv7(conv7)
        conv7 = torch.cat([conv7,x11],1)
        
        x20 = self.up_sample_Dense_2(conv7)
#        print('size of X_Dense_11 :', x11.size())           #  ([1, 512, 128, 128])
        x21 = self.up_sample_Dense_21(conv7)                               
#        print('size of X_Dense_12 :', x12.size())           #  ([1, 512, 256, 256])
        
        
        # Decoder Block 3
        deconv2 = self.deconv2(conv7)
        conv8 = torch.cat((deconv2, conv2), 1)
        conv8 = self.conv8(conv8)
        conv8 = torch.cat([conv8, x12, x20], 1)
        
        x30 = self.up_sample_Dense_3(conv8)

        # Decoder Block 4
        
        deconv1 = self.deconv1(conv8)
        conv9 = torch.cat((deconv1, conv1), 1)
        conv9 = self.conv9(conv9)
        conv9 = torch.cat([conv9, x13, x21, x30], 1)

        output = self.conv10(conv9)

        return output
    
    
class DUNetLAM(nn.Module):    
    def __init__(self, band_num, class_num):
        super(DUNetLAM, self).__init__()
        self.band_num = band_num
        self.class_num = class_num
        self.name = 'DUNetLAM'

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

        self.deconv4 = DUpsampling(channels[4], scale_factor = 2, class_num=15)
        self.conv6 = nn.Sequential(
            conv3otherRelu(channels[4]-241, channels[3]),
            conv3otherRelu(channels[3], channels[3]),
        )

        self.deconv3 = DUpsampling(channels[3], scale_factor = 2, class_num=15)
        self.conv7 = nn.Sequential(
            conv3otherRelu(channels[3]-113, channels[2]),
            conv3otherRelu(channels[2], channels[2]),
        )

        self.deconv2 = DUpsampling(channels[2], scale_factor = 2, class_num=15)
        self.conv8 = nn.Sequential(
            conv3otherRelu(channels[2]-49, channels[1], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            conv3otherRelu(channels[1], channels[1], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )

        self.deconv1 = DUpsampling(channels[1], scale_factor = 2, class_num=15)
        self.conv9 = nn.Sequential(
            conv3otherRelu(channels[1]-17, channels[0], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            conv3otherRelu(channels[0], channels[0], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )
        
        self.lpa = PositionLinearAttention(channels[0])
        self.lca = ChannelLinearAttention()

        self.conv10 = nn.Conv2d(channels[0], self.class_num, kernel_size=1, stride=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        deconv4 = self.deconv4(conv5)
        # print('size of DECONV4 :', deconv4.size())
        conv6 = torch.cat((deconv4, conv4), 1)
        conv6 = self.conv6(conv6)

        deconv3 = self.deconv3(conv6)
        conv7 = torch.cat((deconv3, conv3), 1)
        conv7 = self.conv7(conv7)

        deconv2 = self.deconv2(conv7)
        conv8 = torch.cat((deconv2, conv2), 1)
        conv8 = self.conv8(conv8)

        deconv1 = self.deconv1(conv8)
        conv9 = torch.cat((deconv1, conv1), 1)
        conv9 = self.conv9(conv9)
        
        lpa = self.lpa(conv9)

        lca = self.lca(conv9)

        feat_sum = lpa + lca

        output = self.conv10(feat_sum)

        #output = self.conv10(conv9)

        return output
    
class DUNetLAMCM(nn.Module):    
    def __init__(self, band_num, class_num):
        super(DUNetLAMCM, self).__init__()
        self.band_num = band_num
        self.class_num = class_num
        self.name = 'DUNetLAMCM'

        channels = [64, 128, 256, 512, 1024]
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

        self.deconv4 = DUpsampling(channels[4], scale_factor = 2, class_num=15)
        self.conv6 = nn.Sequential(
            conv3otherRelu(channels[4]-497, channels[3]),
            conv3otherRelu(channels[3], channels[3]),
        )

        self.deconv3 = DUpsampling(channels[3], scale_factor = 2, class_num=15)
        self.conv7 = nn.Sequential(
            conv3otherRelu(channels[3]-241, channels[2]),
            conv3otherRelu(channels[2], channels[2]),
        )

        self.deconv2 = DUpsampling(channels[2], scale_factor = 2, class_num=15)
        self.conv8 = nn.Sequential(
            conv3otherRelu(channels[2]-113, channels[1], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            conv3otherRelu(channels[1], channels[1], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )

        self.deconv1 = DUpsampling(channels[1], scale_factor = 2, class_num=15)
        self.conv9 = nn.Sequential(
            conv3otherRelu(channels[1]-49, channels[0], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            conv3otherRelu(channels[0], channels[0], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )
        
        self.lpa = PositionLinearAttention(channels[0])
        self.lca = ChannelLinearAttention()

        self.conv10 = nn.Conv2d(channels[0], self.class_num, kernel_size=1, stride=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        deconv4 = self.deconv4(conv5)
        # print('size of DECONV4 :', deconv4.size())
        conv6 = torch.cat((deconv4, conv4), 1)
        conv6 = self.conv6(conv6)

        deconv3 = self.deconv3(conv6)
        conv7 = torch.cat((deconv3, conv3), 1)
        conv7 = self.conv7(conv7)

        deconv2 = self.deconv2(conv7)
        conv8 = torch.cat((deconv2, conv2), 1)
        conv8 = self.conv8(conv8)

        deconv1 = self.deconv1(conv8)
        conv9 = torch.cat((deconv1, conv1), 1)
        conv9 = self.conv9(conv9)
        
        lpa = self.lpa(conv9)

        lca = self.lca(conv9)

        feat_sum = lpa + lca

        output = self.conv10(feat_sum)

        #output = self.conv10(conv9)

        return output
    
    
class DUNetDense(nn.Module):
    def __init__(self, band_num, class_num):
        super(DUNetDense, self).__init__()
        self.band_num = band_num
        self.class_num = class_num
        self.name = 'DUNetDense-5C'

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

        self.deconv4 = DUpsampling(channels[4], scale_factor = 2, class_num=15)
        self.conv6 = nn.Sequential(
            conv3otherRelu(channels[4]-241, channels[3]),
            conv3otherRelu(channels[3], channels[3]),
        )
        
        self.up_sample_Dense_11 = DUpsampling(
                                        channels[3],
                                        scale_factor = 2, 
                                        class_num=15)
        
        self.up_sample_Dense_12 = DUpsampling(
                                        channels[3],
                                        scale_factor = 4, 
                                        class_num=15)
        
        self.up_sample_Dense_13 = DUpsampling(
                                        channels[3],
                                        scale_factor = 8, 
                                        class_num=15)

        self.deconv3 = DUpsampling(channels[3], scale_factor = 2, class_num=15)
        self.conv7 = nn.Sequential(
            conv3otherRelu(channels[3]-113, channels[2]),
            conv3otherRelu(channels[2], channels[2]),
        )
        
        self.up_sample_Dense_2 = DUpsampling(
                                        channels[2]+15,
                                        scale_factor = 2, 
                                        class_num=15)
        
        self.up_sample_Dense_21 = DUpsampling(
                                        channels[2]+15,
                                        scale_factor = 4, 
                                        class_num=15)

        self.deconv2 = DUpsampling(channels[2]+15, scale_factor = 2, class_num=15)
        self.conv8 = nn.Sequential(
            conv3otherRelu(channels[2]-49, channels[1], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            conv3otherRelu(channels[1], channels[1], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )
        
        self.up_sample_Dense_3 = DUpsampling(
                                        channels[1]+30,
                                        scale_factor = 2, 
                                        class_num=15)

        self.deconv1 = DUpsampling(channels[1]+30, scale_factor = 2, class_num=15)
        self.conv9 = nn.Sequential(
            conv3otherRelu(channels[1]-17, channels[0], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            conv3otherRelu(channels[0], channels[0], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )

        self.conv10 = nn.Conv2d(channels[0]+45, self.class_num, kernel_size=1, stride=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        #print('size of X_E_1 :', conv1.size())
        conv2 = self.conv2(conv1)
        #print('size of X_E_2 :', conv2.size())
        conv3 = self.conv3(conv2)
        #print('size of X_E_3 :', conv3.size())
        conv4 = self.conv4(conv3)
        #print('size of X_E_4 :', conv4.size())
        conv5 = self.conv5(conv4)
        #print('size of X_E_5 :', conv5.size())


        # Decoder Block 1        
        deconv4 = self.deconv4(conv5)
        #print('size of X_D_1 :', deconv4.size())
        conv6 = torch.cat((deconv4, conv4), 1)
        conv6 = self.conv6(conv6)                            #  ([32, 32])
        #print('size of X_DC_1 :', conv6.size())
        
        x11 = self.up_sample_Dense_11(conv6)
        #print('size of X_Dense_11 :', x11.size())           #  ([64, 64])
        x12 = self.up_sample_Dense_12(conv6)                               
        #print('size of X_Dense_12 :', x12.size())           #  ([128, 128])
        x13 = self.up_sample_Dense_13(conv6)                               
        #print('size of X_Dense_13 :', x13.size())           #  ([256, 256])

        # Decoder Block 2
        deconv3 = self.deconv3(conv6)
        #print('size of X_D_2 :', x.size())
        conv7 = torch.cat((deconv3, conv3), 1)
        conv7 = self.conv7(conv7)
        #print('size of X_DC_2 :', conv7.size())
        conv7 = torch.cat([conv7, x11], 1)
        #print('size of X_DC_11_Addup :', conv7.size())
        
        x20 = self.up_sample_Dense_2(conv7)
        #print('size of X_Dense_11 :', x11.size())           #  ([64, 64])
        x21 = self.up_sample_Dense_21(conv7)                               
        #print('size of X_Dense_12 :', x12.size())           #  ([128, 128])
        
        
        # Decoder Block 3
        deconv2 = self.deconv2(conv7)
        #print('size of X_DC_11_Addup :', deconv2.size())
        conv8 = torch.cat((deconv2, conv2), 1)
        conv8 = self.conv8(conv8)
        #print('size of X_DC_11_Addup :', conv8.size())
        conv8 = torch.cat([conv8, x12, x20], 1)
        #print('size of X_DC_11_Addup :', conv8.size())
        
        x30 = self.up_sample_Dense_3(conv8)
        #print('size of X_DC_3_Addup :', x30.size())

        # Decoder Block 4 
        deconv1 = self.deconv1(conv8)
        #print('size of X_DC_11_Addup :', deconv1.size())
        conv9 = torch.cat((deconv1, conv1), 1)
        conv9 = self.conv9(conv9)
        #print('size of X_DC_11_Addup :', conv9.size())
        conv9 = torch.cat([conv9, x13, x21, x30], 1)
        #print('size of X_DC_11_Addup :', conv9.size())

        output = self.conv10(conv9)

        return output
    
class DUNetDenseLAM(nn.Module):
    def __init__(self, band_num, class_num):
        super(DUNetDenseLAM, self).__init__()
        self.band_num = band_num
        self.class_num = class_num
        self.name = 'DUNetDenseLAM'

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

        self.deconv4 = DUpsampling(channels[4], scale_factor = 2, class_num=15)
        self.conv6 = nn.Sequential(
            conv3otherRelu(channels[4]-241, channels[3]),
            conv3otherRelu(channels[3], channels[3]),
        )
        
        self.up_sample_Dense_11 = DUpsampling(
                                        channels[3],
                                        scale_factor = 2, 
                                        class_num=15)
        
        self.up_sample_Dense_12 = DUpsampling(
                                        channels[3],
                                        scale_factor = 4, 
                                        class_num=15)
        
        self.up_sample_Dense_13 = DUpsampling(
                                        channels[3],
                                        scale_factor = 8, 
                                        class_num=15)

        self.deconv3 = DUpsampling(channels[3], scale_factor = 2, class_num=15)
        self.conv7 = nn.Sequential(
            conv3otherRelu(channels[3]-113, channels[2]),
            conv3otherRelu(channels[2], channels[2]),
        )
        
        self.up_sample_Dense_2 = DUpsampling(
                                        channels[2]+15,
                                        scale_factor = 2, 
                                        class_num=15)
        
        self.up_sample_Dense_21 = DUpsampling(
                                        channels[2]+15,
                                        scale_factor = 4, 
                                        class_num=15)

        self.deconv2 = DUpsampling(channels[2]+15, scale_factor = 2, class_num=15)
        self.conv8 = nn.Sequential(
            conv3otherRelu(channels[2]-49, channels[1], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            conv3otherRelu(channels[1], channels[1], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )
        
        self.up_sample_Dense_3 = DUpsampling(
                                        channels[1]+30,
                                        scale_factor = 2, 
                                        class_num=15)

        self.deconv1 = DUpsampling(channels[1]+30, scale_factor = 2, class_num=15)
        self.conv9 = nn.Sequential(
            conv3otherRelu(channels[1]-17, channels[0], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            conv3otherRelu(channels[0], channels[0], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )
        
        self.lpa = PositionLinearAttention(channels[0] + 45)
        self.lca = ChannelLinearAttention()

        self.conv10 = nn.Conv2d(channels[0]+45, self.class_num, kernel_size=1, stride=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        #print('size of X_E_1 :', conv1.size())
        conv2 = self.conv2(conv1)
        #print('size of X_E_2 :', conv2.size())
        conv3 = self.conv3(conv2)
        #print('size of X_E_3 :', conv3.size())
        conv4 = self.conv4(conv3)
        #print('size of X_E_4 :', conv4.size())
        conv5 = self.conv5(conv4)
        #print('size of X_E_5 :', conv5.size())


        # Decoder Block 1        
        deconv4 = self.deconv4(conv5)
        #print('size of X_D_1 :', deconv4.size())
        conv6 = torch.cat((deconv4, conv4), 1)
        conv6 = self.conv6(conv6)                            #  ([32, 32])
        #print('size of X_DC_1 :', conv6.size())
        
        x11 = self.up_sample_Dense_11(conv6)
        #print('size of X_Dense_11 :', x11.size())           #  ([64, 64])
        x12 = self.up_sample_Dense_12(conv6)                               
        #print('size of X_Dense_12 :', x12.size())           #  ([128, 128])
        x13 = self.up_sample_Dense_13(conv6)                               
        #print('size of X_Dense_13 :', x13.size())           #  ([256, 256])

        # Decoder Block 2
        deconv3 = self.deconv3(conv6)
        #print('size of X_D_2 :', x.size())
        conv7 = torch.cat((deconv3, conv3), 1)
        conv7 = self.conv7(conv7)
        #print('size of X_DC_2 :', conv7.size())
        conv7 = torch.cat([conv7, x11], 1)
        #print('size of X_DC_11_Addup :', conv7.size())
        
        x20 = self.up_sample_Dense_2(conv7)
        #print('size of X_Dense_11 :', x11.size())           #  ([64, 64])
        x21 = self.up_sample_Dense_21(conv7)                               
        #print('size of X_Dense_12 :', x12.size())           #  ([128, 128])
        
        
        # Decoder Block 3
        deconv2 = self.deconv2(conv7)
        #print('size of X_DC_11_Addup :', deconv2.size())
        conv8 = torch.cat((deconv2, conv2), 1)
        conv8 = self.conv8(conv8)
        #print('size of X_DC_11_Addup :', conv8.size())
        conv8 = torch.cat([conv8, x12, x20], 1)
        #print('size of X_DC_11_Addup :', conv8.size())
        
        x30 = self.up_sample_Dense_3(conv8)
        #print('size of X_DC_3_Addup :', x30.size())

        # Decoder Block 4 
        deconv1 = self.deconv1(conv8)
        #print('size of X_DC_11_Addup :', deconv1.size())
        conv9 = torch.cat((deconv1, conv1), 1)
        conv9 = self.conv9(conv9)
        #print('size of X_DC_11_Addup :', conv9.size())
        conv9 = torch.cat([conv9, x13, x21, x30], 1)
        #print('size of X_DC_11_Addup :', conv9.size())
        
        lpa = self.lpa(conv9)

        lca = self.lca(conv9)

        feat_sum = lpa + lca

        output = self.conv10(feat_sum)

        # output = self.conv10(conv9)

        return output
    
class DUNetDenseLAMfour(nn.Module):
    def __init__(self, band_num, class_num):
        super(DUNetDenseLAMfour, self).__init__()
        self.band_num = band_num
        self.class_num = class_num
        # self.name = 'DUNetDenseLAMfour-BR-6C'
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
        
        self.T = torch.nn.Parameter(torch.Tensor([1.00]))

    def forward(self, x):
        conv1 = self.conv1(x)
#        print('size of X_E_1 :', conv1.size())
        conv2 = self.conv2(conv1)
#        print('size of X_E_2 :', conv2.size())
        conv3 = self.conv3(conv2)
#        print('size of X_E_3 :', conv3.size())
        conv4 = self.conv4(conv3)
#        print('size of X_E_4 :', conv4.size())
        conv5 = self.conv5(conv4)
#        print('size of X_E_5 :', conv5.size())

        # Decoder Block 1        
        deconv4 = self.deconv4(conv5)
        deconv4 = deconv4/self.T
#        print('size of X_D_1 :', deconv4.size())
        conv6 = torch.cat((deconv4, conv4), 1)
        
        lpa = self.lpa4(conv6)
        lca = self.lca4(conv6)
        feat_sum = lpa + lca
        conv6 = self.conv6(feat_sum)
        
        #conv6 = self.conv6(conv6)                            #  ([32, 32])
#        print('size of X_DC_1 :', conv6.size())
        
        x11 = self.up_sample_Dense_11(conv6)
#        print('size of X_Dense_11 :', x11.size())           #  ([64, 64])
        x12 = self.up_sample_Dense_12(conv6)                               
#        print('size of X_Dense_12 :', x12.size())           #  ([128, 128])
        x13 = self.up_sample_Dense_13(conv6)                               
#        print('size of X_Dense_13 :', x13.size())           #  ([256, 256])

        # Decoder Block 2
        deconv3 = self.deconv3(conv6)
        deconv3 = deconv3 / self.T
#        print('size of X_D_2 :', x.size())
        conv7 = torch.cat((deconv3, conv3), 1)
        conv7 = self.conv7(conv7)
#        print('size of X_DC_2 :', conv7.size())
        conv7 = torch.cat([conv7, x11], 1)
#        print('size of X_DC_11_Addup :', conv7.size())
        
        lpa = self.lpa3(conv7)
        lca = self.lca3(conv7)
        feat_sum = lpa + lca
        conv7 = self.conv7(feat_sum)
        #print('size of X_DC_2 :', conv7.size())

        x20 = self.up_sample_Dense_2(conv7)
#        print('size of X_Dense_11 :', x11.size())           #  ([64, 64])
        x21 = self.up_sample_Dense_21(conv7)                               
#        print('size of X_Dense_12 :', x12.size())           #  ([128, 128])
              
        # Decoder Block 3
        deconv2 = self.deconv2(conv7)
        deconv2 = deconv2 / self.T
#        print('size of X_DC_11_Addup :', deconv2.size())
        conv8 = torch.cat((deconv2, conv2), 1)
        conv8 = self.conv8(conv8)
#        print('size of X_DC_11_Addup :', conv8.size())
        conv8 = torch.cat([conv8, x12, x20], 1)
#        print('size of X_DC_11_Addup :', conv8.size())
        
        lpa = self.lpa2(conv8)
        lca = self.lca2(conv8)
        feat_sum = lpa + lca
        conv8 = self.conv81(feat_sum)
        
        x30 = self.up_sample_Dense_3(conv8)
#        print('size of X_DC_3_Addup :', x30.size())

        # Decoder Block 4 
        deconv1 = self.deconv1(conv8)
        deconv1 = deconv1 / self.T
#        print('size of X_DC_11_Addup :', deconv1.size())
        conv9 = torch.cat((deconv1, conv1), 1)
        conv9 = self.conv9(conv9)
#        print('size of X_DC_11_Addup :', conv9.size())
        conv9 = torch.cat([conv9, x13, x21, x30], 1)
#        print('size of X_DC_11_Addup :', conv9.size())
        
        lpa = self.lpa1(conv9)
        lca = self.lca1(conv9)
        feat_sum = lpa + lca

        output = self.conv10(feat_sum)
        # out = output / self.T

        # output = self.conv10(conv9)

        return output
    
class DUNetDenseLinearAttention(nn.Module):
    def __init__(self, band_num, class_num):
        super(DUNetDenseLinearAttention, self).__init__()
        self.band_num = band_num
        self.class_num = class_num
        self.name = 'DUNetDenseLinearAttention'

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
        
        self.LA4 = LinearAttention(channels[4])
        # self.lpa4 = PositionLinearAttention(channels[4])
        # self.lca4 = ChannelLinearAttention()

        self.deconv3 = DUpsampling(channels[3], scale_factor = 2, class_num=5)
        self.conv7 = nn.Sequential(
            conv3otherRelu(channels[3], channels[2]),
            conv3otherRelu(channels[2], channels[2]),
        )
        
        self.LA3 = LinearAttention(channels[3])
        # self.lpa3 = PositionLinearAttention(channels[3])
        # self.lca3 = ChannelLinearAttention()
        
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
        
        
        self.LA2 = LinearAttention(channels[2] + 64)
        # self.lpa2 = PositionLinearAttention(channels[2]+64)
        # self.lca2 = ChannelLinearAttention()
        
        self.up_sample_Dense_3 = DUpsampling(
                                        channels[1],
                                        scale_factor = 2, 
                                        class_num=5)

        self.deconv1 = DUpsampling(channels[1], scale_factor = 2, class_num=5)
        self.conv9 = nn.Sequential(
            conv3otherRelu(channels[1], channels[0], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            conv3otherRelu(channels[0], channels[0], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )
        
        self.LA1 = LinearAttention(channels[2])
        # self.lpa1 = PositionLinearAttention(channels[2])
        # self.lca1 = ChannelLinearAttention()

        self.conv10 = nn.Conv2d(channels[2], self.class_num, kernel_size=1, stride=1)
        
        self.T = torch.nn.Parameter(torch.Tensor([1.00]))

    def forward(self, x):
        conv1 = self.conv1(x)
#        print('size of X_E_1 :', conv1.size())
        conv2 = self.conv2(conv1)
#        print('size of X_E_2 :', conv2.size())
        conv3 = self.conv3(conv2)
#        print('size of X_E_3 :', conv3.size())
        conv4 = self.conv4(conv3)
#        print('size of X_E_4 :', conv4.size())
        conv5 = self.conv5(conv4)
#        print('size of X_E_5 :', conv5.size())

        # Decoder Block 1        
        deconv4 = self.deconv4(conv5)
        deconv4 = deconv4/self.T
#        print('size of X_D_1 :', deconv4.size())
        conv6 = torch.cat((deconv4, conv4), 1)
        
        # lpa = self.lpa4(conv6)
        # lca = self.lca4(conv6)
        feat_sum = self.LA4(conv6)
        conv6 = self.conv6(feat_sum)
        
        #conv6 = self.conv6(conv6)                            #  ([32, 32])
#        print('size of X_DC_1 :', conv6.size())
        
        x11 = self.up_sample_Dense_11(conv6)
#        print('size of X_Dense_11 :', x11.size())           #  ([64, 64])
        x12 = self.up_sample_Dense_12(conv6)                               
#        print('size of X_Dense_12 :', x12.size())           #  ([128, 128])
        x13 = self.up_sample_Dense_13(conv6)                               
#        print('size of X_Dense_13 :', x13.size())           #  ([256, 256])

        # Decoder Block 2
        deconv3 = self.deconv3(conv6)
        deconv3 = deconv3 / self.T
#        print('size of X_D_2 :', x.size())
        conv7 = torch.cat((deconv3, conv3), 1)
        conv7 = self.conv7(conv7)
#        print('size of X_DC_2 :', conv7.size())
        conv7 = torch.cat([conv7, x11], 1)
#        print('size of X_DC_11_Addup :', conv7.size())
        
        # lpa = self.lpa3(conv7)
        # lca = self.lca3(conv7)
        feat_sum =self.LA3(conv7)
        conv7 = self.conv7(feat_sum)
        #print('size of X_DC_2 :', conv7.size())

        x20 = self.up_sample_Dense_2(conv7)
#        print('size of X_Dense_11 :', x11.size())           #  ([64, 64])
        x21 = self.up_sample_Dense_21(conv7)                               
#        print('size of X_Dense_12 :', x12.size())           #  ([128, 128])
              
        # Decoder Block 3
        deconv2 = self.deconv2(conv7)
        deconv2 = deconv2 / self.T
#        print('size of X_DC_11_Addup :', deconv2.size())
        conv8 = torch.cat((deconv2, conv2), 1)
        conv8 = self.conv8(conv8)
#        print('size of X_DC_11_Addup :', conv8.size())
        conv8 = torch.cat([conv8, x12, x20], 1)
#        print('size of X_DC_11_Addup :', conv8.size())
        
        # lpa = self.lpa2(conv8)
        # lca = self.lca2(conv8)
        feat_sum = self.LA2(conv8)
        conv8 = self.conv81(feat_sum)
        
        x30 = self.up_sample_Dense_3(conv8)
#        print('size of X_DC_3_Addup :', x30.size())

        # Decoder Block 4 
        deconv1 = self.deconv1(conv8)
        deconv1 = deconv1 / self.T
#        print('size of X_DC_11_Addup :', deconv1.size())
        conv9 = torch.cat((deconv1, conv1), 1)
        conv9 = self.conv9(conv9)
#        print('size of X_DC_11_Addup :', conv9.size())
        conv9 = torch.cat([conv9, x13, x21, x30], 1)
#        print('size of X_DC_11_Addup :', conv9.size())
        
        # lpa = self.lpa1(conv9)
        # lca = self.lca1(conv9)
        feat_sum = self.LA1(conv9)

        output = self.conv10(feat_sum)
        # out = output / self.T

        # output = self.conv10(conv9)

        return output