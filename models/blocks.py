import torch
import torch.nn as nn
import torch.nn.functional as F

import os
from matplotlib import pyplot as plt

class MRCB(nn.Module):
    """Multi-region contrast block"""
    def __init__(self, n_feats):
        super(MRCB, self).__init__()
        self.n_feats = n_feats

        self.diff_conv4 = nn.Sequential(  # H//4
            nn.Conv2d(n_feats, n_feats, 3, 1, 1),
            nn.ReLU(inplace=True)
        )
        self.diff_conv8 = nn.Sequential(  # H//8
            nn.Conv2d(n_feats, n_feats, 3, 1, 1),
            nn.ReLU(inplace=True)
        )
        self.diff_conv16 = nn.Sequential(  # H//16
            nn.Conv2d(n_feats, n_feats, 3, 1, 1),
            nn.ReLU(inplace=True)
        )
        self.diff_conv32 = nn.Sequential(  # H//32
            nn.Conv2d(n_feats, n_feats, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

        self.fuse_conv = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, 1, 1),
            nn.LeakyReLU(0.2, True)
        )

        self.conv_proj = nn.Sequential(
            nn.Conv2d(n_feats * 2, n_feats, 3, 1, 1),
            nn.LeakyReLU(0.2, True)
        )

    def forward(self, feat_ori, feat_ahe):
        B, C, H, W = feat_ori.shape

        def extract_diff(scale_h, scale_w, conv_block):
            pooled_ori = F.adaptive_avg_pool2d(feat_ori, (scale_h, scale_w))
            pooled_ahe = F.adaptive_avg_pool2d(feat_ahe, (scale_h, scale_w))
            diff = pooled_ori - pooled_ahe
            diff = conv_block(diff)
            diff_up = F.interpolate(diff, size=(H, W), mode='bilinear', align_corners=False)
            return diff_up

        diff4 = extract_diff(H // 4, W // 4, self.diff_conv4)
        diff8 = extract_diff(H // 8, W // 8, self.diff_conv8)
        diff16 = extract_diff(H // 16, W // 16, self.diff_conv16)
        diff32 = extract_diff(H // 32, W // 32, self.diff_conv32)

        diff_fused = self.fuse_conv(diff4 + diff8 + diff16 + diff32)

        out = self.conv_proj(torch.cat([feat_ori, diff_fused], dim=1))
        return out


##############--- Co-direction fusion block (CODF) ---################
class CODF(nn.Module):
    def __init__(self, n_feats=16):
        super(CODF, self).__init__()
        self.fb_linear = nn.Linear(n_feats, n_feats, bias=False)
        self.dwconv = nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1, groups=n_feats)  # depth-wise conv
        self.pwconv = nn.Conv2d(n_feats, n_feats, kernel_size=1)  # point-wise conv

    def forward(self, xf, xb):
        # xf, xb: B, H, W, C
        xfb = xf + xb
        xfb_proj = self.fb_linear(xfb)

        xfb_proj = xfb_proj.permute(0, 3, 1, 2).contiguous()
        xfb_spatial = self.dwconv(xfb_proj)
        xfb_spatial = self.pwconv(xfb_spatial)

        xfb_spatial = xfb_spatial.permute(0, 2, 3, 1).contiguous()
        out = xfb + xfb_spatial
        return out

##############--- Cross-Direction fusion block (CRDF) ---################
class CRDF(nn.Module):
    def __init__(self, n_feats=16):
        super(CRDF, self).__init__()
        self.vh_linear1 = nn.Linear(2 * n_feats, n_feats, bias=False)
        self.v_linear = nn.Linear(n_feats, n_feats, bias=False)
        self.h_linear = nn.Linear(n_feats, n_feats, bias=False)
        self.vh_linear2 = nn.Linear(n_feats, n_feats, bias=False)

    def forward(self, xv, xh):
        # b h w c
        xvh = self.vh_linear1(torch.cat([xv, xh], dim=3))

        xvl = self.v_linear(xv * xvh)  # b h w c
        xhl = self.h_linear(xh * xvh)  # b h w c
        out_x = self.vh_linear2(xvl + xhl)  # b h w c

        return out_x

class CGF(nn.Module):
    r"""Cross-guidance fusion block"""
    def __init__(self, n_feats):
        super().__init__()
        self.cat_conv = nn.Sequential(
            nn.Conv2d(2 * n_feats, n_feats, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(n_feats, n_feats, 3, 1, 1)
        )
        self.xo_gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Sigmoid()
        )
        self.xa_gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Sigmoid()
        )
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(n_feats, n_feats, 3, 1, 1)
        )

        self.depth_attn = nn.Sequential(
            nn.Conv2d(1, n_feats // 2, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(n_feats // 2, n_feats, 3, 1, 1),
            nn.Sigmoid()
        )

        self.out_conv = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, xo, xa, xd):
        xd = F.interpolate(xd, size=[xo.size(2), xo.size(3)], mode='nearest')
        x_cat = self.cat_conv(torch.cat([xo, xa], dim=1))
        xo = self.xo_gap(xo) * x_cat + xo
        xa = self.xa_gap(xa) * x_cat + xa
        fuse_x = self.fuse_conv(xo + xa)
        d_att = self.depth_attn(xd)
        out_x = self.out_conv(fuse_x * d_att) + fuse_x
        return out_x


class MSLP(nn.Module):
    r"""Multi-scale local perception block"""

    def __init__(self, n_feats):
        super().__init__()
        self.r_conv = nn.Sequential(
            nn.Conv2d(n_feats, n_feats//4, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(n_feats//4, n_feats//4, 3, 1, 1)
        )

        self.conv_1 = nn.Sequential(
            nn.Conv2d(n_feats // 4, n_feats // 4, 1, 1, 0),
            nn.LeakyReLU(0.2, True),
        )
        self.conv_3 = nn.Sequential(
            nn.Conv2d(n_feats // 4, n_feats // 4, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
        )
        self.conv_5 = nn.Sequential(
            nn.Conv2d(n_feats // 4, n_feats // 4, 5, 1, 2),
            nn.LeakyReLU(0.2, True),
        )
        self.conv_7 = nn.Sequential(
            nn.Conv2d(n_feats // 4, n_feats // 4, 7, 1, 3),
            nn.LeakyReLU(0.2, True),
        )

        self.sig = nn.Sigmoid()

        self.cat_conv = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(n_feats, n_feats, 3, 1, 1)
        )

    def show_feature_map(self, feature_map, path):
        feature_map = feature_map.squeeze(0)
        feature_map = feature_map.view(1, feature_map.shape[0], feature_map.shape[1],
                                       feature_map.shape[2])
        upsample = torch.nn.UpsamplingBilinear2d(size=(256, 256))
        feature_map = upsample(feature_map)
        feature_map = feature_map.view(feature_map.shape[1], feature_map.shape[2], feature_map.shape[3])
        feature_map_num = feature_map.shape[0]
        for index in range(1, feature_map_num + 1):
            plt.axis('off')
            folder_name = path
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
            plt.imsave(folder_name + '//' + str(index) + ".png",
                       feature_map[index - 1].detach().cpu().numpy(), cmap='jet')

    def forward(self, x):
        xr = self.r_conv(x)

        x1 = self.conv_1(xr)
        x3 = self.conv_3(xr)
        x5 = self.conv_5(xr)
        x7 = self.conv_7(xr)

        xr = self.sig(xr)
        x1 = xr * x1
        x3 = xr * x3
        x5 = xr * x5
        x7 = xr * x7
        # self.show_feature_map(x1, 'Attention/x1')
        # self.show_feature_map(x3, 'Attention/x3')
        # self.show_feature_map(x5, 'Attention/x5')
        # self.show_feature_map(x7, 'Attention/x7')
        # self.show_feature_map(torch.cat([x1, x3, x5, x7], dim=1), 'Attention/outx')
        out_x = self.cat_conv(torch.cat([x1, x3, x5, x7], dim=1))

        return out_x
