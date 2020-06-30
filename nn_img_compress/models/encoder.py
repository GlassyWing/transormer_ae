import math
from abc import ABC

import torch
import torch.nn as nn
from torch.nn import init

from .decoder import DecoderLayer


class StateEncoder(nn.Module, ABC):

    def __init__(self, d_model, d_inner,
                 n_sep=10,
                 n_layers=1,
                 drop_p=0, n_head=8, d_head=16,
                 dropatt=0.0, pre_lnorm=False,
                 n_probe=16):
        super().__init__()

        self.n_sep = n_sep
        self.n_probe = n_probe
        self.d_model = d_model
        self.n_layers = n_layers

        self.probe = nn.Parameter(torch.FloatTensor(self.n_probe, self.d_model))

        encoders = []
        for i in range(n_layers):
            encoders.append(DecoderLayer(n_head=n_head, d_model=d_model, d_head=d_head, d_inner=d_inner,
                                         dropout=drop_p,
                                         dropatt=dropatt,
                                         pre_lnorm=pre_lnorm))

        self.encoders = nn.ModuleList(encoders)
        self.offset_linear = nn.Linear(d_model, d_model)
        self._init_params()

    def _init_params(self):
        stdv = 1.0 / math.sqrt(self.d_model)
        self.probe = init.uniform_(self.probe, -stdv, stdv)

    def _conv(self, probe_emb, visible_coor_emb, visible_feat_emb, idx):
        bsz, n_points, d_model = visible_coor_emb.shape

        # (bsz * n_s, n_points / n_s, d_model)
        visible_coor_emb = visible_coor_emb.reshape(bsz * self.n_sep, -1, self.d_model)
        visible_feat_emb = visible_feat_emb.reshape(bsz * self.n_sep, -1, self.d_model)

        # (bsz * n_s, n_probe / n_s, d_model)
        probe_emb = probe_emb.reshape(bsz * self.n_sep, -1, self.d_model)

        # (bsz * n_s, n_probe / n_s, d_model)
        value_emb = self.encoders[idx](probe_emb, visible_coor_emb, visible_feat_emb)

        # (bsz, n_probe, d_model)
        probe_emb = probe_emb.view(bsz, -1, self.d_model)
        value_emb = value_emb.view(bsz, -1, self.d_model)

        return probe_emb, value_emb

    def forward(self, visible_coor_emb, visible_feat_emb):
        """

        :param visible_coor_emb: visible coor embedding, shape of (bsz, n_points, d_model)
        :param visible_feat_emb: visible feat embedding, shpae of (bsz, n_points, d_model)
        :return:
        """

        # (bsz, n_probe, d_model)
        probe_emb = self.probe.unsqueeze(0).expand(visible_coor_emb.shape[0], -1, -1)
        probe_emb, value_emb = self._conv(probe_emb, visible_coor_emb, visible_feat_emb, 0)

        probe_embs = [probe_emb]
        value_embs = [value_emb]

        for i in range(1, len(self.encoders)):
            offset = self.offset_linear(value_emb)
            probe_emb = probe_emb + offset
            _, value_emb = self._conv(probe_emb, visible_coor_emb, visible_feat_emb, i)
            probe_embs.append(probe_emb)
            value_embs.append(value_emb)

        probe_embs = torch.cat(probe_embs, dim=1)
        value_embs = torch.cat(value_embs, dim=1)

        # 将长度缩放至1
        # value_emb = value_emb / torch.norm(value_emb, dim=-1, keepdim=True)

        return probe_embs, value_embs


if __name__ == '__main__':
    inputs = torch.rand(size=(1, 30, 128))
    coors = torch.rand(size=(1, 30, 128))
    probe = torch.rand(size=(1, 300, 128))
    lengths = torch.tensor([15], dtype=torch.long)
    encoder = StateEncoder(300, d_model=128, d_inner=64)
    hidden_state = encoder(coors, inputs, probe, lengths)
    print(hidden_state.shape)
