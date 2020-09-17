import torch
import torch.nn as nn
import torch.nn.functional as F
from data_util import config
from train_utils import get_cuda

"""
输入：
s_t: [batch_size, hidden_dim]
prev_s: None / [batch_size, t-1, hidden_dim]

输出：
ct_d: [batch_size, hidden_dim]
prev_s: [batch_size, t, hidden_dim]

"""
class decoder_attention(nn.Module):

    def __init__(self):
        super(decoder_attention, self).__init__()

        if config.intra_decoder:
            self.W_prev = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
            self.W_s = nn.Linear(config.hidden_dim, config.hidden_dim)
            self.v = nn.Linear(config.hidden_dim, 1, bias=False)

    def forward(self, s_t, prev_s):

        if config.intra_decoder is False:
            ct_d = get_cuda(torch.zeros(s_t.size()))
        elif prev_s is None:
            ct_d = get_cuda(torch.zeros(s_t.size()))
            prev_s = s_t.unsqueeze(1) # [bs, 1, hid]
        else:
            et = self.W_prev(prev_s)    # [bs, t-1, hid]
            dec_feature = self.W_s(s_t).unsqueeze(1)
            et = et + dec_feature
            et = torch.tanh(et)
            et = self.v(et).squeeze(2)
            at = F.softmax(et, dim=1).unsqueeze(1) # [bs, 1, t-1]

            ct_d = torch.bmm(at, prev_s).squeeze(1) # [bs, hid]

            prev_s = torch.cat([prev_s, s_t.unsqueeze(1)], dim=1) # [bs, t, hid]

        return ct_d, prev_s
