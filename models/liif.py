import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models import register
from utils import make_coord


@register('liif')
class LIIF(nn.Module):

    def __init__(self, encoder_spec, imnet_spec=None,
                 local_ensemble=False, feat_unfold=True, cell_decode=False):
        super().__init__()
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode

        self.encoder = models.make(encoder_spec)
        # self.unsample = nn.UpsamplingBilinear2d
        self.deconv_H2_W2 = torch.nn.ConvTranspose2d(64, 64, kernel_size=(1, 4), stride=(1, 2),
                                                 padding=(0, 1), output_padding=(0, 0))
        self.deconv_H4_W4 = torch.nn.ConvTranspose2d(64, 64, kernel_size=(1, 8), stride=(1, 4),
                                                 padding=(0, 2), output_padding=(0, 0))
        self.shrinking = nn.Conv2d(64 * 3, 64, kernel_size=1, stride=1)


        # self.skip_conv = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(3, 1), padding=(0, 1))
        # self.conv = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1)
        if imnet_spec is not None:
            imnet_in_dim = 64
            if self.feat_unfold:
                imnet_in_dim *= 9
            imnet_in_dim += 2 # attach coord
            if self.cell_decode:
                imnet_in_dim += 2
            self.imnet = models.make(imnet_spec, args={'in_dim': imnet_in_dim})
        else:
            self.imnet = None

    def gen_feat(self, inp):
        self.feat = self.encoder(inp, None)
        return self.feat

    def gen_feat_H2_W2(self, inp_down_H2_W2, muti_line_H2_W2):
        self.feat_H2_W2 = self.encoder(inp_down_H2_W2, muti_line_H2_W2)
        return self.feat_H2_W2

    def gen_feat_H4_W4(self, inp_down_H4_W4, muti_line_H4_W4):
        self.feat_H4_W4 = self.encoder(inp_down_H4_W4, muti_line_H4_W4)
        return self.feat_H4_W4

    def query_rgb(self, coord, cell=None):
        feat = self.feat

        if self.feat_unfold:
            feat = F.unfold(feat, 3, padding=1).view(
                feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])

        vx_lst, vy_lst, eps_shift = [0], [0], 0

        # field radius (global: [-1, 1])
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])

        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                q_feat = F.grid_sample(
                    feat, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                q_coord = F.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]
                inp = torch.cat([q_feat, rel_coord], dim=-1)

                bs, q = coord.shape[:2]
                pred = self.imnet(inp.view(bs * q, -1)).view(bs, q, -1)
                preds.append(pred)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)

        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)
        return ret

    def forward(self, inp, coord, cell, muti_line):
        self.gen_feat(inp)
        return self.query_rgb(coord, cell)
