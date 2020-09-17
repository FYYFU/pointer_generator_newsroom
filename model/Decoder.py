import torch
import torch.nn as nn
import torch.nn.functional as F

from model.Encoder_Attention import encoder_attention
from model.Decoder_Attnetion import decoder_attention

from train_utils import get_cuda
from data_util import config, init_wt

"""
LSTM 和 LSTMCell的区别在于：
 LSTM需要参数中提供seq_len,从而能够按照seq_lens的长度来输出每个Cell的隐藏状态。
 而LSTMCell就是一个LSTM单元，不需要提供长度，所以需要手动处理每个时刻的迭代计算过程。
 LSTM: x: [bs, seq_lens, emb]   h,c: [bs, hid]
 LSTMCell: x: [bs, emb]   h,c: [bs, hid]
 
输入：
x_t        [batch_size, emb_dim]                     当前time step的输入
h_t,(h, c) [batch_size, hidden_dim]                  当前time step的hidden_state + cell state。
enc_out,   [batch_size, max_seq_len, 2 * hidden_dim] encoder的所有time step的输出。
enc_padding_mask.   [batch_size, max_seq_len]        encoder输入的padding图，0代表填充，1代表没有填充。
ct_e,               [batch_size, hidden_dim]         作为输入时之前的time step对encoder进行注意力计算得到的结果。
extra_zeros,        [batch_size, batch.max_art_oovs] 用于存储out-of-vocabulary。
enc_batch_extend_vocab [batch_size, max_seq_len]     enc_batch 使用了oovs id。
sum_temporal_srcs      [batch_size, max_seq_len]     存储decoder之前的time step对encoder的每个单元的注意力。
prev_s                 [batch_size,t-1, hidden_dim]  存储decoder之前的time step的预测结果。

输出：
final_dist：       [batch_size, config.vocab_size + batch.max_art_oovs]  当前计算得到
h_t: (h, c):       [batch_size, hidden_dim]       当前部分LSTMCell的输出得到
ct_e:              [batch_size, 2 * hidden_dim]。 encoder_attention得到，当前的time step对encoder进行注意力计算得到的结果。
sum_temporal_srcs: [batch_size, max_seq_len]      encoder_attention得到
prev_s:            [batch_size, t,  hidden_dim]   decoder_attention得到
"""
class Decoder(nn.Module):

    def __init__(self):

        super(Decoder, self).__init__()
        self.encoder_attention = encoder_attention()
        self.decoder_attention = decoder_attention()

        # ct_e + s_t
        self.x_input = nn.Linear(config.hidden_dim * 2 + config.emb_dim, config.emb_dim)

        self.lstm = nn.LSTMCell(config.emb_dim, config.hidden_dim)
        init_wt.init_lstm_wt(self.lstm)

        # x_t, st_hat ct_e, ct_d
        # emb    2      2     1
        self.p_gen_linear = nn.Linear(config.hidden_dim * 5 + config.emb_dim, 1)

        self.V = nn.Linear(config.hidden_dim * 4, config.hidden_dim)
        self.V1 = nn.Linear(config.hidden_dim, config.vocab_size)
        init_wt.init_linear_wt(self.V1)

    def forward(self, x_t, h_t, enc_out, enc_padding_mask, ct_e, extra_zeros,
                enc_batch_extend_vocab, sum_temporal_srcs, prev_s):

        # 这个初始化传入的 ct_e 就是encoder传给decoder的hidden state。
        # 这个 hidden_state 应该就是将 h 和c 组合到一起了。
        # 那么这个ct_e 的size应该是： [batch_size, 2 * hidden_dim]
        x = self.x_input(torch.cat([x_t, ct_e], dim=1))

        # x: [batch_size, emb_dim]
        # s_t: (h, c) [batch_size, hidden_dim]
        h_t = self.lstm(x, h_t)

        dec_h, dec_c = h_t

        # 这里其实有一点问题就是，LSTM应该是单纯的将隐藏状态作为当前time step的输出
        # 但是作者在这里输出的LSTM单元的隐藏状态 h 和记忆状态 c.
        # 原因： 这个st_hat 其实并不是当前time step的输出，输入依旧是 dec_h.这个st_hat主要就是用来计算当前的decoder time step对encoder进行
        #        attention操作以后得到的context向量。
        st_hat = torch.cat([dec_h, dec_c], dim=1)

        ct_e, attn_dist, sum_temporal_srcs = self.encoder_attention(st_hat, enc_out,
                                                                    enc_padding_mask, sum_temporal_srcs)

        ct_d, prev_s = self.decoder_attention(dec_h, prev_s)

        # 在这里还是有一个问题就是：
        # x是当前time step的输入。作者在前面将 x_t 和前面的time step对encoder的注意力向量链接到了一起。
        # 上面的那个连接操作在原始的论文中并没有体现出来。
        p_gen = torch.cat([ct_e, ct_d, st_hat, x], dim=1)

        p_gen = self.p_gen_linear(p_gen)
        p_gen = torch.sigmoid(p_gen)

        out = torch.cat([dec_h, ct_e, ct_d], dim=1)
        out = self.V(out)
        out = self.V1(out)
        vocab_dist = F.softmax(out, dim=1)
        vocab_dist = p_gen * vocab_dist
        attn_dist_ = (1 - p_gen) * attn_dist

        if extra_zeros is not None:
            # extra_zeros: [batch_size, max_art_oovs]
            vocab_dist = torch.cat([vocab_dist, extra_zeros], dim=1)
        # self_tensor: vocab_dist
        # index_tensor: enc_batch_extend_vocab
        # other_tensor: attn_dist_
        # 根据index_tensor 将 other_tensor加入到self_tensor中。
        # vocab_dist[i][enc_batch_extend_vocab[i][j]] += attn_dist_[i][j]

        final_dist = vocab_dist.scatter_add(1, enc_batch_extend_vocab, attn_dist_)

        return final_dist, h_t, ct_e, sum_temporal_srcs, prev_s
