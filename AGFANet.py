from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from pointnet2_ops import pointnet2_utils
from torch.nn import init


class SEAttention(nn.Module):
    def __init__(self, channel=512):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, 64, bias=False),
            nn.GELU(),
            nn.Linear(64, channel, bias=False),
            nn.Sigmoid()
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x)
        y = y.view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


def gather_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


class transformer(nn.Module):
    def __init__(self, d_ori=256, d_out=256, head=4, d_fed=1024):
        super().__init__()
        self.multihead_attention1 = nn.MultiheadAttention(d_out, head)
        self.linear1 = nn.Linear(d_out, d_fed)
        self.linear2 = nn.Linear(d_fed, d_out)
        self.norm1 = nn.LayerNorm(d_out)
        self.norm2 = nn.LayerNorm(d_out)
        self.act1 = torch.nn.GELU()
        self.transfer = nn.Conv1d(d_ori, d_out, kernel_size=1)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, q, kv):
        q = self.transfer(q)
        kv = self.transfer(kv)
        b, c, _ = q.shape
        q = q.reshape(b, c, -1).permute(2, 0, 1)
        kv = kv.reshape(b, c, -1).permute(2, 0, 1)
        q = self.norm2(q)
        kv = self.norm2(kv)
        qq = self.multihead_attention1(query=q,
                                     key=kv,
                                     value=kv)[0]
        q = q + qq
        q = self.norm1(q)
        qq = self.linear2(self.act1(self.linear1(q)))
        q = q + qq
        q = q.permute(1, 2, 0)
        return q


class transformer_cross(nn.Module):
    def __init__(self, d_ori=256, d_out=256, head=4, d_fed=1024):
        super().__init__()
        self.multihead_attention1 = nn.MultiheadAttention(d_out, head)
        self.linear1 = nn.Linear(d_out, d_fed)
        self.linear2 = nn.Linear(d_fed, d_out)
        self.norm1 = nn.LayerNorm(d_out)
        self.norm2 = nn.LayerNorm(d_out)
        self.norm3 = nn.LayerNorm(d_out)
        self.act1 = torch.nn.GELU()
        self.transfer1 = nn.Conv1d(d_ori, d_out, kernel_size=1)
        self.transfer2 = nn.Conv1d(d_ori, d_out, kernel_size=1)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, q, kv):
        q = self.transfer1(q)
        kv = self.transfer2(kv)
        b, c, _ = q.shape
        q = q.reshape(b, c, -1).permute(2, 0, 1)
        kv = kv.reshape(b, c, -1).permute(2, 0, 1)
        q = self.norm2(q)
        kv = self.norm3(kv)
        qq = self.multihead_attention1(query=q,
                                     key=kv,
                                     value=kv)[0]
        q = q + qq
        q = self.norm1(q)
        qq = self.linear2(self.act1(self.linear1(q)))
        q = q + qq
        q = q.permute(1, 2, 0)
        return q


class refiner(nn.Module):
    def __init__(self, channel=128, ratio=1, channel_ahead=64):
        super(refiner, self).__init__()
        self.ratio = ratio
        self.conv_1 = nn.Conv1d(256, channel, kernel_size=1)
        self.conv_11 = nn.Conv1d(512, 256, kernel_size=1)
        self.conv_x = nn.Conv1d(3, 64, kernel_size=1)
        self.sa1 = transformer_cross(channel * 2, 256)
        self.sa11 = transformer(channel * 2, 256)
        self.sa2 = transformer(256, 256)
        self.sa3 = transformer(256, channel * ratio)
        self.relu = nn.GELU()
        self.conv_out = nn.Conv1d(64, 3, kernel_size=1)
        self.ps_2 = nn.ConvTranspose1d(channel * 2, channel, 2, 2, bias=True)
        self.channel = channel
        self.conv_delta = nn.Conv1d(channel * 2, channel * 1, kernel_size=1)
        self.conv_ps = nn.Conv1d(channel * ratio, channel * ratio, kernel_size=1)
        self.conv_x1 = nn.Conv1d(64, channel, kernel_size=1)
        self.seatten_1 = SEAttention(channel=256)
        self.conv_out1 = nn.Conv1d(channel, 64, kernel_size=1)
        self.transform = nn.Conv1d(channel_ahead, 256, kernel_size=1)

    def forward(self, x, coarse, feat_g, x_ahead):
        batch_size, _, N = coarse.size()
        y = self.conv_x1(self.relu(self.conv_x(coarse)))
        feat_g = self.conv_1(self.relu(self.conv_11(feat_g)))
        y0 = torch.cat([y, feat_g.repeat(1, 1, y.shape[-1])], dim=1)
        x_ahead = self.relu(self.transform(x_ahead))
        y1 = self.sa1(x_ahead, y0) + self.sa11(y0, y0)
        y2 = self.sa2(y1, y1)
        y2 = self.seatten_1(y2)
        y2 = self.sa3(y2, y2)
        # y2 = self.relu(self.ps_2(self.conv_ps(y2)))
        y2 = self.conv_ps(y2).reshape(batch_size, -1, N * self.ratio)
        y_up = y.repeat(1, 1, self.ratio)
        y_cat = torch.cat([y2, y_up], dim=1)
        y2 = self.conv_delta(y_cat)
        x = self.conv_out(self.relu(self.conv_out1(y2))) + coarse.repeat(1, 1, self.ratio)
        return x, y2


class extractor(nn.Module):
    def __init__(self, channel=64):
        super(extractor, self).__init__()
        self.channel = channel
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 32, kernel_size=1)
        self.sa0 = transformer(channel // 2, channel // 2)
        self.sa0_1 = transformer(channel, channel)
        self.sa0_2 = transformer(channel, channel)
        self.sa1 = transformer(channel, channel)
        self.sa1_1 = transformer(channel * 2, channel * 2)
        self.sa1_2 = transformer(channel * 2, channel * 2)
        self.sa2 = transformer(channel * 2, channel * 2)
        self.sa2_1 = transformer(channel * 4, channel * 4)
        self.sa2_2 = transformer(channel * 4, channel * 4)
        self.sa3 = transformer(channel * 4, channel * 4)
        self.sa3_1 = transformer(channel * 8, channel * 8)
        self.sa3_2 = transformer(channel * 8, channel * 8)
        self.relu = nn.GELU()
        self.sa0_d = transformer_cross(channel * 8, channel * 8)
        self.sa00_d = transformer(channel * 8, channel * 8)
        self.sa1_d = transformer(channel * 8, channel * 8)
        self.sa2_d = transformer(channel * 8, channel * 8)
        self.conv_out = nn.Conv1d(64, 3, kernel_size=1)
        self.conv_out1 = nn.Conv1d(channel * 2, 64, kernel_size=1)
        self.ps_1 = nn.ConvTranspose1d(channel * 8, channel, 128, bias=True)
        self.ps_2 = nn.ConvTranspose1d(512, 128, 4, 4, bias=True)
        self.ps_refuse = nn.Conv1d(channel, channel * 8, kernel_size=1)
        self.ps_adj = nn.Conv1d(channel * 8, channel * 8, kernel_size=1)
        self.transform = nn.Conv1d(512, 512, kernel_size=1)
        self.seatten_1 = SEAttention(channel=256)
        self.seatten_2 = SEAttention(channel=512)

    def forward(self, points):
        batch_size, _, N = points.size()
        x = self.relu(self.conv1(points))
        x0 = self.conv2(x)

        idx_0 = pointnet2_utils.furthest_point_sample(points.transpose(1, 2).contiguous(), N // 2).long()
        x_g0 = gather_points(x0.transpose(1, 2).contiguous(), idx_0).transpose(1, 2).contiguous()
        points = gather_points(points.transpose(1, 2).contiguous(), idx_0).transpose(1, 2).contiguous()
        x1 = self.sa0(x_g0, x0).contiguous()
        x1 = torch.cat([x_g0, x1], dim=1)
        x1 = self.sa0_1(x1, x1).contiguous()
        # x1 = self.sa0_2(x1, x1).contiguous()

        idx_1 = pointnet2_utils.furthest_point_sample(points.transpose(1, 2).contiguous(), N // 4).long()
        x_g1 = gather_points(x1.transpose(1, 2).contiguous(), idx_1).transpose(1, 2).contiguous()
        points = gather_points(points.transpose(1, 2).contiguous(), idx_1).transpose(1, 2).contiguous()
        x2 = self.sa1(x_g1, x1).contiguous()  # C*2, N
        x2 = torch.cat([x_g1, x2], dim=1)
        x2 = self.sa1_1(x2, x2).contiguous()
        # x2 = self.sa1_2(x2, x2).contiguous()

        idx_2 = pointnet2_utils.furthest_point_sample(points.transpose(1, 2).contiguous(), N // 8).long()
        x_g2 = gather_points(x2.transpose(1, 2).contiguous(), idx_2).transpose(1, 2).contiguous()
        points = gather_points(points.transpose(1, 2).contiguous(), idx_2).transpose(1, 2).contiguous()
        x3 = self.sa2(x_g2, x2).contiguous()
        x3 = torch.cat([x_g2, x3], dim=1)
        x3 = self.sa2_1(x3, x3).contiguous()
        x3 = self.seatten_1(x3)
        # x3 = self.sa2_2(x3, x3).contiguous()

        idx_3 = pointnet2_utils.furthest_point_sample(points.transpose(1, 2).contiguous(), N // 16).long()
        x_g3 = gather_points(x3.transpose(1, 2).contiguous(), idx_3).transpose(1, 2).contiguous()
        # points = gather_points(points.transpose(1, 2).contiguous(), idx_3).transpose(1, 2).contiguous()
        x4 = self.sa3(x_g3, x3).contiguous()
        x4 = torch.cat([x_g3, x4], dim=1)
        x4 = self.sa3_1(x4, x4).contiguous()
        x4 = self.seatten_2(x4)
        # x4 = self.sa3_2(x4, x4).contiguous()

        x_g = F.adaptive_max_pool1d(x4, 1).view(batch_size, -1).unsqueeze(-1)
        x = self.relu(self.ps_adj(x_g))
        x = self.relu(self.ps_1(x))
        x = self.relu(self.ps_refuse(x))
        x4 = self.relu(self.transform(x4))
        x0_d = (self.sa0_d(x4, x)) + (self.sa00_d(x, x))
        x1_d = (self.sa1_d(x0_d, x0_d))
        # x2_d = (self.sa2_d(x1_d, x1_d)).reshape(batch_size,self.channel*2,N//4)
        # x2_d = self.relu(self.ps_2(self.sa2_d(x1_d, x1_d)))
        x2_d = self.sa2_d(x1_d, x1_d).reshape(batch_size, 128, 512)
        fine = self.conv_out(self.relu(self.conv_out1(x2_d)))
        return x_g, fine, x2_d


class AGFANet(nn.Module):
    def __init__(self):
        super(AGFANet, self).__init__()
        self.encoder = extractor()
        self.refine1 = refiner(ratio=4, channel_ahead=128)
        self.refine2 = refiner(ratio=8, channel_ahead=128)

    def forward(self, x):
        x = x.transpose(1, 2).contiguous()
        feat_g, coarse, x1 = self.encoder(x)
        new_x = torch.cat([x, coarse], dim=2)
        new_x = gather_points(new_x.transpose(1, 2).contiguous(),
                              pointnet2_utils.furthest_point_sample(new_x.transpose(1, 2).contiguous(),
                                                                    512).long()).transpose(1, 2).contiguous()
        fine, feat_fine = self.refine1(None, new_x, feat_g, x1)
        fine1, feat_fine1 = self.refine2(feat_fine, fine, feat_g, feat_fine)
        coarse = coarse.transpose(1, 2).contiguous()
        new_x = new_x.transpose(1, 2).contiguous()
        fine = fine.transpose(1, 2).contiguous()
        fine1 = fine1.transpose(1, 2).contiguous()
        return new_x, fine, fine1


if __name__ == '__main__':
    data = torch.rand(16, 2048, 3).cuda()
    gt = torch.rand(16, 16384, 3).cuda()
    model = AGFANet().cuda()
    out = model(data)
    print(out[2].shape)