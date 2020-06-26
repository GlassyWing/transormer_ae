import time
from abc import ABC

import torch
import torch.nn as nn

from nn_img_compress.models.common import Sine, Swish
from .decoder import StateDecoder
from .encoder import StateEncoder


class StateDAE(nn.Module, ABC):

    def __init__(self, coor_size, feat_size,
                 out_size,
                 n_probe=1,
                 n_sep=1,
                 n_enc_layers=1,
                 n_dec_layers=1,
                 d_model=128, d_inner=64,
                 drop_p=0.0, n_head=8, d_head=16,
                 dropatt=0.0, pre_lnorm=False):
        super().__init__()
        self.coor_size = coor_size
        self.feat_size = feat_size
        self.d_inner = d_inner
        self.d_model = d_model

        self.coor_mapping = nn.Sequential(
            nn.Linear(in_features=coor_size, out_features=d_inner),
            nn.Tanh(),
            nn.Linear(in_features=d_inner, out_features=d_model),
        )

        self.feat_mapping = nn.Sequential(
            nn.Linear(in_features=feat_size, out_features=d_inner),
            nn.Tanh(),
            nn.Linear(in_features=d_inner, out_features=d_model),
        )

        self.state_encoder = StateEncoder(n_probe=n_probe, n_sep=n_sep, n_layers=n_enc_layers,
                                          d_model=d_model, d_inner=d_inner, drop_p=drop_p,
                                          n_head=n_head, d_head=d_head, dropatt=dropatt, pre_lnorm=pre_lnorm)

        self.state_decoder = StateDecoder(n_probe=n_probe, n_layers=n_dec_layers,
                                          d_model=d_model, d_inner=d_inner, drop_p=drop_p,
                                          n_head=n_head, d_head=d_head, dropatt=dropatt, pre_lnorm=pre_lnorm)

        self.o_net = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=d_inner),
            nn.Tanh(),
            nn.Linear(in_features=d_inner, out_features=out_size)
        )

    def forward(self, visible_coor, visible_feat, coor):
        """

        :param visible_coor: shape of (bsz, n_points, coor_size)
        :param visible_feat: shape of (bsz, n_points, feat_size)
        :param coor: shape of (bsz, d_n_points, coor_size)
        :return:
        """

        coor = self.coor_mapping(coor)
        visible_coor = self.coor_mapping(visible_coor)
        visible_feat = self.feat_mapping(visible_feat)

        # inputs = visible_coor + visible_feat

        # Encode
        # (bsz, n_probe, d_model)
        probe_emb, encode_feat = self.state_encoder(visible_coor, visible_feat)

        # (bsz, d_n_points, d_model)
        decode_feat = self.state_decoder(coor, probe_emb, encode_feat)

        return self.o_net(decode_feat)


if __name__ == '__main__':
    n_point = 120
    visible_coor = torch.rand(size=(32, n_point, 3))
    visible_feat = torch.rand(size=(32, n_point, 10))
    coor = torch.rand(size=(32, 420, 3))
    feat = torch.rand(size=(32, 420, 10))
    lengths = torch.ones(size=(32,), dtype=torch.long) * n_point
    o_lengths = torch.ones(size=(32,), dtype=torch.long) * n_point

    state_ae = StateDAE(coor_size=3, feat_size=10, out_size=10,
                        n_probe=32,
                        n_dec_layers=2,
                        n_enc_layers=2,
                        n_sep=4, )

    start_time = time.time()
    outputs = state_ae(visible_coor, visible_feat, lengths, coor)
    print(f"cost {(time.time() - start_time)}s")
    print(outputs.shape)

    # img = torch.rand(size=(1, 10, 28, 28))
    #
    # denoising_ae = DenoisingAutoencoder(28, 28, 10)
    #
    # start_time = time.time()
    # output = denoising_ae(img)
    # print(f"cost {(time.time() - start_time)}s")
    # print(output.shape)
