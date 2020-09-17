import torch
import torch.nn as nn
import torch.nn.functional as F
from data_util import config, init_wt
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()

        self.lstm = nn.LSTM(config.emb_dim, config.hidden_dim, num_layers=1, bidirectional=True)
        init_wt.init_lstm_wt(lstm=self.lstm)

        self.reduce_h = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        init_wt.init_linear_wt(self.reduce_h)

        self.reduce_c = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        init_wt.init_linear_wt(self.reduce_c)

    def forward(self, x, seq_lens):
        """
        :param
               x:     [batch_size, seq_lens, emb_dim]
        :param
            seq_lens: [batch_size]

        :return:
             enc_out: [batch_size, max_seq_len, num_direction * hidden_dim]
             enc_h:   [batch_size, hidden_dim]
             enc_c:   [batch_size, hidden_dim]
        """
        packed = pack_padded_sequence(x, seq_lens, batch_first=True)
        enc_out, enc_hid = self.lstm(packed)
        enc_out, _ = pad_packed_sequence(enc_out, batch_first=True)
        enc_out = enc_out.contiguous()

        enc_h, enc_c = enc_hid

        enc_h = torch.cat(list(enc_h), dim=1)
        enc_c = torch.cat(list(enc_c), dim=1)

        enc_h_reduced= F.relu(self.reduce_h(enc_h))
        enc_c_reduced = F.relu(self.reduce_c(enc_c))

        return enc_out, (enc_h_reduced, enc_c_reduced)


# if __name__ == '__main__':
#
#     Encoder = Encoder()
#
#     seq = torch.tensor([[[2, 2, 2], [2, 2, 2], [2, 2, 2]], [[1, 1, 1], [1,1, 1], [0, 0,0]], [[3, 3, 3], [0, 0,0], [0, 0, 0]]])
#     lens = torch.tensor([3, 2, 1])
#
#     print(seq.shape)
#     print(lens.shape)
#     packed = pack_padded_sequence(seq, lens, batch_first=True, enforce_sorted=False)
#
#     enc_out, (enc_h, enc_c) = Encoder(seq, lens)
#
#     print(enc_out.shape)
#     print(enc_h.shape)
#     print(enc_c.shape)







