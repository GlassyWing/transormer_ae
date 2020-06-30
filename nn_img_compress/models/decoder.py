from abc import ABC

from .attn import MultiHeadAttn
from .common import PositionalWiseFF
import torch.nn as nn
import torch


class DecoderLayer(nn.Module):

    def __init__(self, n_head, d_model, d_head, d_inner, dropout, **kwargs):
        super().__init__()

        self.dec_attn = MultiHeadAttn(n_head, d_model, d_head, dropout, **kwargs)
        self.pos_ff = PositionalWiseFF(d_model, d_inner, dropout,
                                       pre_lnorm=kwargs.get('pre_lnorm'))

    def forward(self, q, k=None, v=None, attn_mask=None):
        output = self.dec_attn(q, k, v, attn_mask)
        output = self.pos_ff(output)

        return output


class StateDecoder(nn.Module, ABC):

    def __init__(self, n_probe, d_model, d_inner, drop_p=0, n_layers=1, n_head=8, d_head=16,
                 dropatt=0.0, pre_lnorm=False):
        super().__init__()

        self.n_layers = n_layers

        decoders = []
        for i in range(n_layers):
            decoders.append(DecoderLayer(n_head=n_head, d_model=d_model, d_head=d_head, d_inner=d_inner,
                                         dropout=drop_p,
                                         dropatt=dropatt,
                                         pre_lnorm=pre_lnorm))

        self.decoders = nn.ModuleList(decoders)

    def forward(self, coor_emb, probe_emb, encode_emb):
        """

        :param encode_emb: shape of (bsz, n_probe, d_model)
        :param probe_emb: shape of (n_probe, d_model) or (bsz, n_probe, d_model)
        :param coor_emb: shape of (bsz, n_d_points, d_model)
        :return:
        """

        if probe_emb.dim() == 2:
            probe_emb = probe_emb.unsqueeze(0).repeat(coor_emb.shape[0], 1, 1)

        # (bsz, d_n_points, d_model)
        coor_emb_parts = torch.split(coor_emb, 30000, dim=1)
        value_embs = []
        for coor_emb_part in coor_emb_parts:
            value_emb = self.decoders[0](coor_emb_part, probe_emb, encode_emb)
            value_embs.append(value_emb)

        return torch.cat(value_embs, dim=1)


if __name__ == '__main__':
    inputs = torch.rand(size=(1, 30, 3))
    hidden_state = torch.rand(size=(1, 1, 128))

    state_decoder = StateDecoder(coor_size=3, d_model=128, num_layers=1)

    outputs = state_decoder(inputs, hidden_state)
    print(outputs.shape)
