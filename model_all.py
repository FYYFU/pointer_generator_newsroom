# import torch
# import torch.nn as nn
#
# from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
# import torch.nn.functional as F
#
# from data_util import config
# from train_utils import get_cuda
#
#
# def init_lstm_wt(lstm):
#     for name, _ in lstm.named_parameters():
#         if 'weight' in name:
#             wt = getattr(lstm, name)
#             wt.data.uniform_(-config.rand_unif_init_mag, config.rand_unif_init_mag)
#         elif 'bias' in name:
#             # set forget bias to 1
#             bias = getattr(lstm, name)
#             n = bias.size(0)
#             start, end = n // 4, n // 2
#             bias.data.fill_(0.)
#             bias.data[start:end].fill_(1.)
#
#
# def init_linear_wt(linear):
#     linear.weight.data.normal_(std=config.trunc_norm_init_std)
#     if linear.bias is not None:
#         linear.bias.data.normal_(std=config.trunc_norm_init_std)
#
#
# def init_wt_normal(wt):
#     wt.data.normal_(std=config.trunc_norm_init_std)
#
#
# class Encoder(nn.Module):
#
#     def __init__(self):
#         super(Encoder, self).__init__()
#
#         self.lstm = nn.LSTM(config.emb_dim, config.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
#         init_lstm_wt(self.lstm)
#
#         # 由于encoder 是 bidirectional LSTM，而decoder是 single directional LSTM.
#         # 所以需要对 encoder的hidden_state进行降维处理。
#         self.reduce_h = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
#         init_linear_wt(self.reduce_h)
#
#         self.reduce_c = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
#         init_linear_wt(self.reduce_c)
#
#     def forward(self, x, seq_lens):
#         """
#         :param x: [batch_size, seq_lens, *]
#         :param seq_lens: [article_number]
#
#         返回值：
#
#         """
#         # 返回的packed是一个PackedSequence对象。能够直接用来作为RNN的输入。
#         # 将输入的维度转化为一维，去掉所有的填充的零，然后使用seq_lens来存储每个article的长度。
#         # 并且根据article的seq_len来对输入进行排序。越长越靠前。
#         packed = pack_padded_sequence(x, seq_lens, batch_first=True)
#
#         # 返回值：enc_out: 依旧是一个PackedSequence。需要使用下面的方法进行处理，
#         # 如果传入的不是一个pakedSequence，那么输出的维度应该是(seq_lens, batch_size, num_directions, hidden_size)
#         # enc_hid = (h_n, c_n). 分别代表了最后一个unit的hidden_state以及cell_state
#         enc_out, enc_hid = self.lstm(packed)
#
#         # 是上面的pack_padded_sequence()的逆方法。能够将输出从一维转换为和输入完全相同的维度，并且进行填充。
#         # 返回两个：一个是输出，另一个是batch中的每个article的长度。
#         enc_out, _ = pad_packed_sequence(enc_out, batch_first=True)
#
#         # 使用continguous()是为了保证底层的连续性(默认行优先存储)，
#         # 通过pad_packed_sequence改变了
#         # 原始的enc_output的形状。所以需要使用continguous()来强制拷贝一份新的内存。
#         enc_out = enc_out.contiguous() # [batch_size, seq_lens,2 * hidden_size]
#
#         h, c = enc_hid                 # [2, batch_size, hidden_size]
#         # h 和 c的size都是: [num_layers*num_directions, batch_size, hidden_size]
#         # 讲两个directional的hidden_state 连结到一起。
#         h = torch.cat(list(h), dim=1)  # [batch_size, 2 * hidden_size]
#         c = torch.cat(list(c), dim=1)
#
#         h_reduced = F.relu(self.reduce_h(h))
#         c_reduced = F.relu(self.reduce_c(c))
#
#         return enc_out, (h_reduced, c_reduced)
#
#
# class encoder_attention(nn.Module):
#
#     def __init__(self):
#
#         super(encoder_attention, self).__init__()
#         self.W_h = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2, bias=False)
#         self.W_s = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2)
#         self.v = nn.Linear(config.hidden_dim * 2, 1, bias=False)
#
#     def forward(self, st_hat, h , enc_padding_mask, sum_temporal_srcs):
#         """
#         :param st_hat:
#             当前 time step的decoder hidden_state.包括了 h 和 c。
#                 [batch_size, 2 * hidden_dim]
#         :param h:
#             encoder 的 hidden_state
#         :param enc_padding_mask:
#         :param sum_temporal_srcs:
#             如果使用了intra-attention，用来存储之前的time_step对当前encoder unit的attention weight。
#         """
#
#         # 这个h对应的应该是 encoder中的enc_out。
#         # 这里采样的attention是 加法模型。
#         et = self.W_h(h)                        # [bs, seq_lens, 2 * hidden]
#         dec_fea = self.W_s(st_hat).unsqueeze(1) # [bs, 1, 2 * hidden]
#         et = et + dec_fea
#         et = torch.tanh(et)         # [batch_size, seq_lens, 2 * hidden_size]
#         et = self.v(et).squeeze(2)  # [batch_size, seq_lens]. 转化为每个词的概率。
#
#         # intra_temporal attention
#         if config.intra_encoder:
#             exp_et = torch.exp(et)
#             if sum_temporal_srcs is None:
#                 et1 = exp_et
#                 sum_temporal_srcs = get_cuda(torch.FloatTensor(et.size()).fill_(1e-10)) + exp_et
#             else:
#                 et1 = exp_et / sum_temporal_srcs
#                 sum_temporal_srcs = sum_temporal_srcs + exp_et
#         else:
#             et1 = F.softmax(et, dim=1)
#
#         # 将padding对应的部分的概率置为0.
#         at = et1 * enc_padding_mask # [batch_size, seq_lens]
#         normalization_factor = at.sum(dim=1, keepdim=True)
#         at = at / normalization_factor
#
#         at = at.unsqueeze(1)    # [bs, 1, seq_lens]
#         ct_e = torch.bmm(at, h) # [bs, 1, 2 * hidden_dim]
#         ct_e = ct_e.squeeze(1)
#         at = at.squeeze(1)
#
#         # at: [batch_size, seq_lens]
#         # ct_e: [batch_size, 2 * hidden_dim]
#         # sum_temporal_srcs: [batch_size, seq_lens]
#         return ct_e, at, sum_temporal_srcs
#
#
# class decoder_attention(nn.Module):
#     def __init__(self):
#         super(decoder_attention, self).__init__()
#         if config.intra_decoder:
#             self.W_prev = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
#             self.W_s = nn.Linear(config.hidden_dim, config.hidden_dim)
#             self.v = nn.Linear(config.hidden_dim, 1, bias=False)
#
#     def forward(self, s_t, prev_s):
#
#         if config.intra_decoder is False:
#             ct_d = get_cuda(torch.zeros(s_t.size()))
#         elif prev_s is None:
#             ct_d = get_cuda(torch.zeros(s_t.size()))
#             prev_s = s_t.unsqueeze(1)               #bs, 1, n_hid
#         else:
#             # Standard attention technique (eq 1 in https://arxiv.org/pdf/1704.04368.pdf)
#             et = self.W_prev(prev_s)                # bs,t-1,n_hid
#             dec_fea = self.W_s(s_t).unsqueeze(1)    # bs,1,n_hid
#             et = et + dec_fea
#             et = torch.tanh(et)                         # bs,t-1,n_hid
#             et = self.v(et).squeeze(2)              # bs,t-1
#             # intra-decoder attention     (eq 7 & 8 in https://arxiv.org/pdf/1705.04304.pdf)
#             at = F.softmax(et, dim=1).unsqueeze(1)  #bs, 1, t-1
#             ct_d = torch.bmm(at, prev_s).squeeze(1)     #bs, n_hid
#             prev_s = torch.cat([prev_s, s_t.unsqueeze(1)], dim=1)    #bs, t, n_hid
#
#         # ct_d: [batch_size, hidden_dim]
#         # prev_s: [batch_size, t, hidden_dim]
#         return ct_d, prev_s
#
#
# class Decoder(nn.Module):
#
#     def __init__(self):
#         super(Decoder, self).__init__()
#         self.enc_attention = encoder_attention()
#         self.dec_attention = decoder_attention()
#         # 输入 emb_dim + attention 输出 hidden_dim * 2
#         self.x_context = nn.Linear(config.hidden_dim * 2 + config.emb_dim, config.emb_dim)
#
#         self.lstm = nn.LSTMCell(config.emb_dim, config.hidden_dim)
#         init_lstm_wt(self.lstm)
#
#         # 采用pointer的概率。
#         self.p_gen_linear = nn.Linear(config.hidden_dim * 5 + config.emb_dim, 1)
#
#         self.V = nn.Linear(config.hidden_dim *4, config.hidden_dim)
#         self.V1 = nn.Linear(config.hidden_dim, config.vocab_size)
#
#     def forward(self, x_t, s_t, enc_out, enc_padding_mask, ct_e, extra_zeros,
#                 enc_batch_extend_vocab, sum_temporal_srcs, prev_s):
#
#         # x_t 是 decoder 在time step t的时候的输入 [batch_size, emb_dim]
#         # ct_e应该是从encoder中传给decoder最后的hidden state [batch_size, 2 * hidden_state]
#         x = self.x_context(torch.cat([x_t, ct_e], dim=1))
#
#         # x 的size就是：[batch_size, emb_dim]
#         # s_t 应该就是前面的time step传来的h 和 c。[batch_size, hidden_dim]
#         s_t = self.lstm(x, s_t)
#
#         dec_h, dec_c = s_t
#         st_hat = torch.cat([dec_h, dec_c], dim=1)
#         ct_e, atte_dist, sum_temporal_srcs = self.enc_attention(st_hat, enc_out,
#                                                                 enc_padding_mask, sum_temporal_srcs)
#
#         ct_d, prev_s = self.dec_attention(dec_h, prev_s)
#
#         # 对encoder attention得到的ct_e     [batch_size, 2 * hidden_dim]
#         # 对decoder attention得到的 ct_d    [batch_size, hidden_dim]
#         # decoder当前time step的预测 st_hat [batch_size, 2 * hidden_dim]
#         # decoder当前time step的输入 x      [batch_size, emb_dim]
#         # 得到的 p_gen [batch_size, 1]
#         # 计算得到的 p_gen 是 采用abstractive 方式来生成单词的概率。
#         p_gen = torch.cat([ct_e, ct_d, st_hat, x], 1)
#
#         p_gen = self.p_gen_linear(p_gen)
#         p_gen = torch.sigmoid(p_gen)
#
#         out = torch.cat([dec_h, ct_e, ct_d], dim=1) # [bs, 4*hidden_dims]
#         out = self.V(out)                           # [bs, hidden_dim]
#         out = self.V1(out)                          # [bs, vocab_dim]
#
#         vocab_list = torch.softmax(out, dim=1)
#         vocab_list = p_gen * vocab_list
#         attn_dist = (1 - p_gen) * atte_dist
#
#         if extra_zeros is not None:
#             vocab_list = torch.cat([vocab_list, extra_zeros], dim=1)
#
#         final_dist = vocab_list.scatter_add(1, enc_batch_extend_vocab, attn_dist)
#
#         return final_dist, s_t, ct_e, sum_temporal_srcs, prev_s
#
# class Model(nn.Module):
#
#     def __init__(self):
#         super(Model, self).__init__()
#         self.encoder = Encoder()
#         self.decoder = Decoder()
#         self.embeds = nn.Embedding(config.vocab_size, config.emb_dim)
#         init_wt_normal(self.embeds.weight)
#
#         self.encoder = get_cuda(self.encoder)
#         self.decoder = get_cuda(self.decoder)
#         self.embeds = get_cuda(self.embeds)
#
#
#
#
#
#
#
#
#
#
#
#
#
