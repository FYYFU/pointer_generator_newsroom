import torch
import torch.nn as nn
import torch.nn.functional as F
from data_util import config
from train_utils import get_cuda

"""
输入值:

enc_out: [batch_size, seq_lens, 2 * hidden_dim]
st_hat: [batch_size, 2 * hidden_dim] (h, c)
enc_padding_mask: [batch_size, seq_lens]
sum_temporal_srcs: None / [batch_size, seq_lens]

返回值：
at: [batch_size, max_seq_len]
ct_e: [batch_size, 2 * hidden_dim]
sum_temporal_srcs: [batch_size, seq_lens]
"""
class encoder_attention(nn.Module):

    def __init__(self):

        super(encoder_attention, self).__init__()

        self.W_h = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2, bias=False)
        self.W_s = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2)
        self.v = nn.Linear(config.hidden_dim * 2, 1, bias=False)

    def forward(self, st_hat, enc_out, enc_padding_mask, sum_temporal_srcs):

        et = self.W_h(enc_out)             # [bs, max_seq_len, 2 * hid]
        dec_feature = self.W_s(st_hat).unsqueeze(1)
        et = et + dec_feature
        et = torch.tanh(et)                # [bs, max_seq_len, 2 * hid]
        et = self.v(et).squeeze(2)         # [bs, seq_lens]

        if config.intra_encoder:
            exp_et = torch.exp(et)
            if sum_temporal_srcs is None:
                et1 = exp_et
                sum_temporal_srcs = get_cuda(torch.FloatTensor(et.size()).fill_(1e-10)) + exp_et
            else:
                et1 = exp_et / sum_temporal_srcs
                sum_temporal_srcs = sum_temporal_srcs + exp_et
        else:
            et1 = F.softmax(et, dim=1)

        # 通过mask将padding对应的部分的概率设置为0. element-wise
        at = et1 * enc_padding_mask
        normalization_factor = at.sum(dim=1, keepdim=True)
        at = at / normalization_factor

        at = at.unsqueeze(1)
        ct_e = torch.bmm(at, enc_out)
        ct_e = ct_e.squeeze(1)
        at = at.squeeze(1)

        return ct_e, at, sum_temporal_srcs



# import numpy as np
#
# if __name__ == '__main__':
#     enc_out = torch.from_numpy(np.zeros((2, 55, 1024), dtype=np.int32)).float()
#     st_hat = torch.from_numpy(np.zeros((2, 1024), dtype=np.int32)).float()
#     enc_padding_mask = torch.from_numpy(np.zeros((2, 55), dtype=np.float32)).float()
#     sum_temporal_srcs = None
#
#     attention = encoder_attention()
#
#     ct_e, at, sum_temporal_srcs = attention(st_hat, enc_out, enc_padding_mask, sum_temporal_srcs)
#
#     print(attention)
#
#     print(at.shape)
#     print(ct_e.shape)
#     print(sum_temporal_srcs.shape)
