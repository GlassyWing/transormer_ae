import torch
import torch.nn as nn


def _split(t, n_head, d_head):
    return t.view(t.size(0), t.size(1), n_head, d_head)


class MultiHeadAttn(nn.Module):

    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0, pre_lnorm=True):
        super().__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head

        self.q_net = nn.Linear(d_model, n_head * d_head, bias=False)
        self.k_net = nn.Linear(d_model, n_head * d_head, bias=False)
        self.v_net = nn.Linear(d_model, n_head * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)

        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)
        self.scale = 1. / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm

        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, q, k=None, v=None, attn_mask=None):
        """

        :param q: shape of (bsz, n1, d_model)
        :param k: shape of (bsz, n2, d_model)
        :param v: shape of (bsz, n2, d_model)
        :return: result shape of (bsz, n1, d_model)
        """

        if self.pre_lnorm:
            q = self.layer_norm(q)

        head_q = self.q_net(q)
        head_k = self.k_net(k) if k is not None else self.k_net(q)
        head_v = self.v_net(v) if v is not None else self.v_net(q)

        head_q = _split(head_q, self.n_head, self.d_head)  # (bsz, n1, n_head, d_head)
        head_k = _split(head_k, self.n_head, self.d_head)
        head_v = _split(head_v, self.n_head, self.d_head)

        # (n1, n2, bsz, n_head)
        attn_score = torch.einsum('bind, bjnd->ijbn', head_q, head_k)
        attn_score.mul_(self.scale)

        if attn_mask is not None and attn_mask.any().item():

            # 掩码维度为(klen, qlen)
            if attn_mask.dim() == 2:
                # 注意全为-float('inf')，softmax之后会变成nan
                attn_score.masked_fill_(attn_mask[:, :, None, None], -1e10)
            elif attn_mask.dim() == 3:
                attn_mask = attn_mask.permute(1, 2, 0)
                attn_score.masked_fill_(attn_mask[:, :, :, None], -1e10)
        attn_prob = torch.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        # 乘以归一化因子
        # attn_prob = attn_prob / (1e-10 + torch.sum(attn_prob, dim=1, keepdim=True))

        # (bsz, n1, n_head, d_model)
        attn_vec = torch.einsum('ijbn, bjnd->bind', attn_prob, head_v)
        attn_vec = attn_vec.contiguous().view(attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        # linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            output = q + attn_out
        else:
            output = self.layer_norm(q + attn_out)

        return output
