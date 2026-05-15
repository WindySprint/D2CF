import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F

from models.blocks import CGF
from models.lgsa import LGSA
from models.lgca import LGCA

class CPGNet(nn.Module):

    def __init__(self, n_feats):
        super(CPGNet, self).__init__()
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

        self.cgf3 = CGF(n_feats * 4)
        self.cgf2 = CGF(n_feats * 2)
        self.cgf1 = CGF(n_feats)

        self.lgsa = LGSA(dim=n_feats * 4,
                        d_state=math.ceil(n_feats * 4 / 6),
                        attn_drop=0., drop_path=0.,
                        norm_layer=nn.LayerNorm)

        self.lgca = LGCA(n_feats * 4, 8)

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
            nn.Conv2d(n_feats, 3, 3, 1, 1, bias=True)
        )

    def forward(self, img_ori, img_ahe, img_d):
        oe1 = self.ini_oconv(img_ori)
        oe2 = self.down_oconv1(oe1)
        oe3 = self.down_oconv2(oe2)

        ae1 = self.ini_aconv(img_ahe)
        ae2 = self.down_aconv1(ae1)
        ae3 = self.down_aconv2(ae2)

        skip1 = self.cgf1(oe1, ae1, img_d)
        skip2 = self.cgf2(oe2, ae2, img_d)
        skip = self.cgf3(oe3, ae3, img_d)

        skip = self.lgsa(skip)
        skip = self.lgca(skip)

        d2 = F.interpolate(skip, scale_factor=2, mode='bilinear', align_corners=False)
        d2 = self.up_conv2(d2) + skip2
        d1 = F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=False)
        d1 = self.up_conv1(d1) + skip1

        img_en = self.conv_last(d1)

        return img_en