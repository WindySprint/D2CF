import torch
import torch.nn as nn

class LCA(nn.Module):
    r"""Local Channel Attention"""
    def __init__(self, n_feats, reduction):
        super(LCA, self).__init__()
        self.avgp = nn.AdaptiveAvgPool2d(1)
        self.clrc = nn.Sequential(
            nn.Conv2d(n_feats, n_feats // reduction, 1, bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(n_feats // reduction, n_feats, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        a = self.clrc(self.avgp(x))
        ca = self.sigmoid(a)
        out = x + x * ca
        return out

class GCA(nn.Module):
    r"""Global Channel Attention"""
    def __init__(self):
        super().__init__()
        self.conv_q = nn.Conv1d(1, 1, 3, 1, 1)
        self.conv_k = nn.Conv1d(1, 1, 3, 1, 1)
        self.avgp = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        B, C, H, W = x.shape

        q = k = self.avgp(x).reshape(B, 1, C)
        q = self.conv_q(q).sigmoid()
        k = self.conv_q(k).sigmoid().permute(0, 2, 1)
        q_k = torch.bmm(k, q).reshape(B, -1)
        q_k = q_k.softmax(-1).reshape(B, C, C)

        v = x.permute(0, 2, 3, 1).reshape(B, -1, C)
        att = torch.bmm(v, q_k).permute(0, 2, 1)
        att = att.reshape(B, C, H, W)
        return x * att

class LGCA(nn.Module):
    r"""Local-Global Channel Attention"""
    def __init__(self, n_feats, reduction):
        super().__init__()
        self.global_attention = GCA()
        self.local_attention = LCA(n_feats, reduction)
        self.cat_conv = nn.Sequential(
            nn.Conv2d(n_feats*2, n_feats, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(n_feats, n_feats, 3, 1, 1)
        )

    def forward(self, x):
        x1 = self.global_attention(x)
        x2 = self.local_attention(x)
        output = self.cat_conv(torch.cat((x1, x2), dim=1))

        return output + x