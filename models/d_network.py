import torch.nn as nn
import torch.nn.functional as F

from models.blocks import MRCB

class RCNet(nn.Module):
    r"""Depth Estimation Based on Region-wise Contrast"""
    def __init__(self, n_feats):
        super(RCNet, self).__init__()
        # Encoder_ori
        self.ini_oconv = nn.Sequential(nn.Conv2d(3, n_feats, 3, 1, 1, bias=True),
                                      nn.LeakyReLU(0.2, True),
                                      nn.Conv2d(n_feats, n_feats, 3, 1, 1, bias=True))

        self.down_oconv1 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(n_feats, n_feats * 2, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, True)
        )
        self.down_oconv2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(n_feats * 2, n_feats * 4, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, True)
        )

        # Encoder_ahe
        self.ini_aconv = nn.Sequential(nn.Conv2d(3, n_feats, 3, 1, 1, bias=True),
                                      nn.LeakyReLU(0.2, True),
                                      nn.Conv2d(n_feats, n_feats, 3, 1, 1, bias=True))

        self.down_aconv1 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(n_feats, n_feats * 2, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, True)
        )
        self.down_aconv2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(n_feats * 2, n_feats * 4, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, True)
        )

        self.mrcb3 = MRCB(n_feats * 4)
        self.mrcb2 = MRCB(n_feats * 2)
        self.mrcb1 = MRCB(n_feats)

        # Decoder
        self.up_conv2 = nn.Sequential(
            nn.Conv2d(n_feats * 4, n_feats * 2, 3, 1, 1),
            nn.LeakyReLU(0.2, True)
        )

        self.up_conv1 = nn.Sequential(
            nn.Conv2d(n_feats * 2, n_feats, 3, 1, 1),
            nn.LeakyReLU(0.2, True)
        )

        self.conv_last = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(n_feats, 1, 3, 1, 1, bias=True)
        )

    def forward(self, img_ori, img_ahe):
        oe1 = self.ini_oconv(img_ori)
        oe2 = self.down_oconv1(oe1)
        oe3 = self.down_oconv2(oe2)

        ae1 = self.ini_aconv(img_ahe)
        ae2 = self.down_aconv1(ae1)
        ae3 = self.down_aconv2(ae2)

        skip1 = self.mrcb1(oe1, ae1)
        skip2 = self.mrcb2(oe2, ae2)
        skip = self.mrcb3(oe3, ae3)

        d2 = F.interpolate(skip, scale_factor=2, mode='bilinear', align_corners=False)
        d2 = self.up_conv2(d2) + skip2
        d1 = F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=False)
        d1 = self.up_conv1(d1) + skip1

        img_depth = self.conv_last(d1)

        return img_depth